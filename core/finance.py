"""
Precision financial calculations using Decimal for trading applications.

This module provides precise arithmetic for financial quantities to avoid
floating-point rounding errors that can cause incorrect order quantities
or prices in production trading systems.

Usage:
    from core.finance import to_decimal, round_quantity, calculate_risk

    price = to_decimal("45000.1234")
    quantity = round_quantity(calculate_risk(entry, sl, risk_usdt))
"""

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Optional, Union

Quantity = Decimal
Price = Decimal
Money = Decimal

# Precision constants for different financial operations
QUANTITY_PRECISION = Decimal("0.001")  # 3 decimal places for crypto quantities
PRICE_PRECISION = Decimal("0.01")  # 2 decimal places for prices
MONEY_PRECISION = Decimal("0.01")  # 2 decimal places for USDT amounts
PERCENT_PRECISION = Decimal("0.0001")  # 4 decimal places for percentages


def to_decimal(
    value: Union[str, float, int, Decimal, None],
    default: Decimal = Decimal("0"),
) -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: The value to convert (str, float, int, or Decimal)
        default: Default value if conversion fails

    Returns:
        Decimal representation of the value
    """
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return default

    if isinstance(value, str):
        try:
            return Decimal(value)
        except (InvalidOperation, ValueError):
            return default

    return default


def round_quantity(quantity: Decimal) -> float:
    """
    Round quantity to exchange-preferred precision.

    Most crypto exchanges require quantities rounded to 3 decimal places.
    Returns a float for compatibility with exchange APIs.
    """
    try:
        rounded = quantity.quantize(QUANTITY_PRECISION, rounding=ROUND_HALF_UP)
        return float(rounded)
    except (InvalidOperation, ValueError):
        return 0.0


def round_price(price: Decimal) -> float:
    """
    Round price to exchange-preferred precision.

    Returns a float for compatibility with exchange APIs.
    """
    try:
        rounded = price.quantize(PRICE_PRECISION, rounding=ROUND_HALF_UP)
        return float(rounded)
    except (InvalidOperation, ValueError):
        return 0.0


def round_money(amount: Decimal) -> float:
    """
    Round monetary amounts to 2 decimal places.

    Returns a float for compatibility with exchange APIs.
    """
    try:
        rounded = amount.quantize(MONEY_PRECISION, rounding=ROUND_HALF_UP)
        return float(rounded)
    except (InvalidOperation, ValueError):
        return 0.0


def calculate_quantity_from_risk(
    risk_amount: Union[float, Decimal],
    entry_price: Union[float, Decimal],
    stop_loss_price: Union[float, Decimal],
    leverage: int = 1,
) -> float:
    """
    Calculate position quantity based on risk amount.

    Args:
        risk_amount: Amount of USDT to risk
        entry_price: Entry price
        stop_loss_price: Stop loss price
        leverage: Leverage multiplier

    Returns:
        Quantity in base currency (rounded to exchange precision)
    """
    risk = to_decimal(risk_amount)
    entry = to_decimal(entry_price)
    sl = to_decimal(stop_loss_price)

    price_risk = abs(entry - sl)
    if price_risk <= 0:
        return 0.0

    leverage_factor = to_decimal(leverage) if leverage > 0 else Decimal("1")
    quantity = (risk / price_risk) * leverage_factor

    return round_quantity(quantity)


def calculate_risk_amount(
    entry_price: Union[float, Decimal],
    stop_loss_price: Union[float, Decimal],
    quantity: Union[float, Decimal],
) -> Decimal:
    """
    Calculate risk amount in quote currency.

    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        quantity: Position quantity

    Returns:
        Risk amount in quote currency (USDT)
    """
    entry = to_decimal(entry_price)
    sl = to_decimal(stop_loss_price)
    qty = to_decimal(quantity)

    return abs(entry - sl) * qty


def calculate_pnl(
    entry_price: Union[float, Decimal],
    exit_price: Union[float, Decimal],
    quantity: Union[float, Decimal],
    side: str = "LONG",
) -> Decimal:
    """
    Calculate profit/loss for a position.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Position quantity
        side: "LONG" or "SHORT"

    Returns:
        PnL in quote currency (positive = profit, negative = loss)
    """
    entry = to_decimal(entry_price)
    exit = to_decimal(exit_price)
    qty = to_decimal(quantity)

    if side.upper() == "LONG":
        return (exit - entry) * qty
    else:  # SHORT
        return (entry - exit) * qty


def calculate_roi_percentage(
    entry_price: Union[float, Decimal],
    exit_price: Union[float, Decimal],
    side: str = "LONG",
) -> Decimal:
    """
    Calculate ROI percentage for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        side: "LONG" or "SHORT"

    Returns:
        ROI as a decimal (e.g., 0.05 = 5% profit)
    """
    entry = to_decimal(entry_price)
    exit_p = to_decimal(exit_price)

    if entry <= 0:
        return Decimal("0")

    if side.upper() == "LONG":
        return (exit_p - entry) / entry
    else:  # SHORT
        return (entry - exit_p) / entry


def calculate_slippage_bps(
    requested_price: Union[float, Decimal],
    fill_price: Union[float, Decimal],
) -> Decimal:
    """
    Calculate slippage in basis points.

    Args:
        requested_price: Price requested by strategy
        fill_price: Actual fill price

    Returns:
        Slippage in bps (positive = unfavorable)
    """
    requested = to_decimal(requested_price)
    fill = to_decimal(fill_price)

    if requested <= 0:
        return Decimal("0")

    # Slippage = (fill - requested) / requested * 10000
    return ((fill - requested) / requested) * Decimal("10000")


def validate_price(positive: bool = True) -> Decimal:
    """
    Get the minimum valid price (for validation checks).

    Args:
        positive: If True, return minimum positive price

    Returns:
        Minimum price value
    """
    if positive:
        return Decimal("0.00000001")
    return Decimal("0")


def safe_division(
    numerator: Union[float, Decimal],
    denominator: Union[float, Decimal],
    default: Decimal = Decimal("0"),
) -> Decimal:
    """
    Safely divide two numbers, returning default if denominator is zero.
    """
    num = to_decimal(numerator)
    denom = to_decimal(denominator)

    if denom <= 0:
        return default

    try:
        return num / denom
    except (InvalidOperation, ZeroDivisionError):
        return default
