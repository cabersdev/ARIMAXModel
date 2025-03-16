# src/backtesting/backtest.py
import backtrader as bt
import pandas as pd
import numpy as np
from backtrader.analyzers import PyFolio, SharpeRatio, AnnualReturn, DrawDown, TradeAnalyzer
from ..modeling.model import AdaptiveARIMAX
import yaml
from multiprocessing import cpu_count
import pyfolio as pf

class BacktestExecutor:
    """Versione ottimizzata con ereditarietà corretta"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._init_cerebro()

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

class ARIMAStrategy(bt.Strategy):
    """Strategia avanzata ARIMA con gestione dinamica del rischio"""
    
    params = (
        ('risk_per_trade', 0.02),       # % del capitale per trade
        ('atr_multiplier', 3.0),         # Moltiplicatore ATR per stop loss
        ('volatility_window', 21),       # Finestra volatilità storica
        ('rebalance_days', 7),           # Giorni tra i ribilanciamenti
        ('max_leverage', 3.0),           # Leverage massimo permesso
    )

    def __init__(self):
        # Inizializzazione indicatori
        self.signal = self.datas[0].signal
        self.volatility = bt.indicators.ATR(self.data, period=self.p.volatility_window)
        self.momentum = bt.indicators.MACDHisto(self.data)
        self.adaptive_sizer = AdaptivePositionSizer(self.p.risk_per_trade, 
                                                   self.p.max_leverage)
        
        # Stato strategia
        self.trade_count = 0
        self.last_rebalance = 0
        self.order = None

        # Registrazione metriche
        self.add_timer(bt.timer.SESSION_END, monthdays=[1])

    def next(self):
        # Ribilanciamento periodico
        if len(self.data) - self.last_rebalance >= self.p.rebalance_days:
            self.close_all_positions()
            self.last_rebalance = len(self.data)

        # Gestione segnali
        if not self.position:
            self.manage_entries()
        else:
            self.manage_exits()

    def manage_entries(self):
        """Gestione ingressi basata su segnali ARIMA"""
        if self.signal[0] == 1:  # Segnale BUY
            size = self.adaptive_sizer.get_size(
                self.broker.getvalue(),
                self.data.close[0],
                self.volatility[0]
            )
            self.order = self.buy(size=size, exectype=bt.Order.Market)
            
        elif self.signal[0] == -1:  # Segnale SELL
            size = self.adaptive_sizer.get_size(
                self.broker.getvalue(),
                self.data.close[0],
                self.volatility[0]
            )
            self.order = self.sell(size=size, exectype=bt.Order.Market)

        if self.order:
            self.trade_count += 1
            self.set_trailing_stop()

    def manage_exits(self):
        """Gestione uscite dinamiche"""
        if self.signal[0] == 0 or self.momentum.macd < 0:
            self.close()

    def set_trailing_stop(self):
        """Trailing stop dinamico basato su volatilità"""
        stop_price = self.data.close[0] - (self.volatility[0] * self.p.atr_multiplier)
        self.order.addinfo(
            stop=stop_price,
            trailamount=self.volatility[0] * self.p.atr_multiplier
        )

    def close_all_positions(self):
        """Chiusura forzata posizioni al ribilanciamento"""
        for position in self.positions:
            self.close(position)

    def notify_trade(self, trade):
        """Registrazione metrica personalizzata"""
        if trade.isclosed:
            pnl = trade.pnlcomm
            self.log(f'Trade {self.trade_count}: PnL={pnl:.2f}')

class AdaptivePositionSizer:
    """Dimensionamento posizione basato su rischio e volatilità"""
    
    def __init__(self, risk_per_trade, max_leverage):
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage

    def get_size(self, portfolio_value, price, volatility):
        """Calcola size posizione con vincoli di leverage"""
        risk_capital = portfolio_value * self.risk_per_trade
        dollar_volatility = price * volatility
        size = risk_capital / dollar_volatility
        max_size = (portfolio_value * self.max_leverage) / price
        return min(size, max_size)

class BacktestEngine:
    """Motore di backtesting avanzato con funzionalità complete"""
    
    def __init__(self, config_path='configs/backtest_config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.cerebro = bt.Cerebro(
            stdstats=False,
            optreturn=False,
            maxcpus=cpu_count() - 1
        )
        
        self._configure_broker()
        self._add_analyzers()

    def _configure_broker(self):
        """Configurazione parametri broker"""
        self.cerebro.broker.setcash(self.config['initial_capital'])
        self.cerebro.broker.setcommission(
            commission=self.config['commission'],
            margin=self.config.get('margin', 1.0)
        )
        
        if 'slippage' in self.config:
            self.cerebro.broker.set_slippage_perc(
                perc=self.config['slippage']['percentage'],
                slip_open=self.config['slippage'].get('open', True),
                slip_match=self.config['slippage'].get('match', True)
            )

    def _add_analyzers(self):
        """Aggiunta analyzer avanzati"""
        self.cerebro.addanalyzer(PyFolio, _name='pyfolio')
        self.cerebro.addanalyzer(SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        self.cerebro.addanalyzer(AnnualReturn, _name='annual')
        self.cerebro.addanalyzer(DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(TradeAnalyzer, _name='trades')

    def prepare_data(self, raw_data):
        """Preparazione dati per Backtrader con segnali ARIMA"""
        model = AdaptiveARIMAX()
        model.fit(raw_data)
        preds = model.predict(raw_data, horizon=self.config['forecast_horizon'])
        
        signals = pd.Series(
            np.where(
                preds['forecast'] > preds['confidence_interval'][1], 1,
                np.where(
                    preds['forecast'] < preds['confidence_interval'][0], -1, 0
                )
            ),
            index=raw_data.index[-len(preds['forecast']):]
        )
        
        merged_data = raw_data.join(signals.rename('signal'), how='inner')
        return bt.feeds.PandasData(
            dataname=merged_data,
            datetime='Date',
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            signal='signal',
            plot=False
        )

    def run_backtest(self, data, optimization=False):
        """Esecuzione backtest/ottimizzazione"""
        self.cerebro.adddata(self.prepare_data(data))
        
        if optimization:
            self._setup_optimization()
        else:
            self.cerebro.addstrategy(ARIMAStrategy, **self.config['strategy_params'])
        
        results = self.cerebro.run()
        return self._process_results(results[0] if not optimization else results)

    def _setup_optimization(self):
        """Configurazione parametri per ottimizzazione"""
        self.cerebro.optstrategy(
            ARIMAStrategy,
            risk_per_trade=self.config['optimization']['risk_range'],
            atr_multiplier=self.config['optimization']['atr_range'],
            rebalance_days=self.config['optimization']['rebalance_days']
        )

    def _process_results(self, results):
        """Elaborazione risultati avanzata"""
        if isinstance(results, list):  # Ottimizzazione
            return self._process_optimization(results)
        return self._process_single_run(results)

    def _process_single_run(self, result):
        """Elaborazione singolo backtest"""
        pyfolio = result.analyzers.getbyname('pyfolio')
        returns, positions, transactions, _ = pyfolio.get_pf_items()
        
        return {
            'performance': {
                'sharpe': result.analyzers.sharpe.get_analysis(),
                'drawdown': result.analyzers.drawdown.get_analysis(),
                'annual_returns': result.analyzers.annual.get_analysis(),
                'trade_stats': result.analyzers.trades.get_analysis()
            },
            'pyfolio_report': pf.create_returns_tear_sheet(returns)
        }

    def _process_optimization(self, results):
        """Elaborazione risultati ottimizzazione"""
        optimized_params = []
        for result in results:
            sharpe = result.analyzers.sharpe.get_analysis()['sharperatio']
            optimized_params.append({
                'params': result.params,
                'sharpe': sharpe,
                'max_dd': result.analyzers.drawdown.get_analysis()['max']['drawdown']
            })
        
        return sorted(optimized_params, key=lambda x: x['sharpe'], reverse=True)

    def generate_report(self, results, report_type='full'):
        """Generazione report dettagliato"""
        if report_type == 'optimization':
            return self._generate_optimization_report(results)
        return self._generate_full_report(results)

    def _generate_full_report(self, results):
        """Report completo con PyFolio"""
        return pf.create_full_tear_sheet(
            results['returns'],
            positions=results['positions'],
            transactions=results['transactions']
        )

    def _generate_optimization_report(self, results):
        df = pd.DataFrame(results)
        return df.style\
            .background_gradient(subset=['sharpe'], cmap='RdYlGn')\
            .highlight_min(subset=['max_dd'], color='#ffcccc')\
            .format({'sharpe': '{:.2f}', 'max_dd': '{:.2%}'})

if __name__ == "__main__":
    engine = BacktestEngine()
    data = pd.read_csv('data/processed/GOOGL.csv', parse_dates=['Date'])
    
    results = engine.run_backtest(data)
    engine.generate_report(results['pyfolio_report'])
    
    optimization_results = engine.run_backtest(data, optimization=True)
    print(engine.generate_report(optimization_results, 'optimization'))