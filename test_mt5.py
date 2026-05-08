from __future__ import annotations

from core.mt5_connector import MT5Connector


connector = MT5Connector()
connector.initialize()
try:
    print("MT5 Connected")
finally:
    connector.shutdown()
