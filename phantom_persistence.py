"""
Live Phantom Trade persistence (PostgreSQL/SQLite via SQLAlchemy)
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

Base = declarative_base()


class PhantomTradeLive(Base):
    __tablename__ = 'phantom_trades_live'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(24), nullable=False)
    side = Column(String(8), nullable=False)
    entry = Column(Float, nullable=False)
    sl = Column(Float, nullable=False)
    tp = Column(Float, nullable=False)
    signal_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    outcome = Column(String(12))  # win|loss|timeout
    realized_rr = Column(Float)
    pnl_percent = Column(Float)
    exit_reason = Column(String(24))
    strategy_name = Column(String(32), default='trend_pullback')
    was_executed = Column(Boolean, default=False)
    ml_score = Column(Float)
    features = Column(Text)  # json serialized


class PhantomPersistence:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.is_postgres = False
        if not self.database_url:
            self.database_url = f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'candles.db')}"
        else:
            if self.database_url.startswith('postgresql://'):
                self.database_url = self.database_url.replace('postgresql://', 'postgresql+psycopg2://')
            self.is_postgres = True
        self.engine = create_engine(self.database_url, poolclass=NullPool, connect_args={} if self.is_postgres else {"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_trade(self, rec: Dict[str, Any]):
        s = self.Session()
        try:
            pt = PhantomTradeLive(
                symbol=str(rec.get('symbol')),
                side=str(rec.get('side')),
                entry=float(rec.get('entry')),
                sl=float(rec.get('sl')),
                tp=float(rec.get('tp')),
                signal_time=rec.get('signal_time') if isinstance(rec.get('signal_time'), datetime) else datetime.fromisoformat(str(rec.get('signal_time'))),
                exit_time=rec.get('exit_time') if (rec.get('exit_time') and isinstance(rec.get('exit_time'), datetime)) else (datetime.fromisoformat(str(rec.get('exit_time'))) if rec.get('exit_time') else None),
                outcome=str(rec.get('outcome')) if rec.get('outcome') else None,
                realized_rr=float(rec.get('realized_rr', 0.0) or 0.0),
                pnl_percent=float(rec.get('pnl_percent', 0.0) or 0.0),
                exit_reason=str(rec.get('exit_reason')) if rec.get('exit_reason') else None,
                strategy_name=str(rec.get('strategy_name', 'trend_pullback')),
                was_executed=bool(rec.get('was_executed', False)),
                ml_score=float(rec.get('ml_score', 0.0) or 0.0),
                features=str(rec.get('features_json', '{}')),
            )
            s.add(pt)
            s.commit()
        finally:
            s.close()

