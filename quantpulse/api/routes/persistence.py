"""PostgreSQL persistence routes for Portfolios, Optimization History, and Seeding."""

import logging
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from quantpulse.api.schemas import (
    PortfolioCreateRequest,
    PortfolioResponse,
    SeedDatabaseResponse,
)
from quantpulse.db.models import BacktestRecord, OptimizationRecord, Portfolio
from quantpulse.db.seed import seed_database
from quantpulse.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/portfolios", response_model=List[PortfolioResponse])
def list_portfolios(db: Session = Depends(get_db)) -> List[PortfolioResponse]:
    """List all configured portfolios from PostgreSQL."""
    portfolios = db.query(Portfolio).all()
    return [
        PortfolioResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            asset_tickers=p.asset_tickers,
            initial_capital=p.initial_capital,
            created_at=p.created_at.isoformat() if p.created_at else None,
            updated_at=p.updated_at.isoformat() if p.updated_at else None,
        )
        for p in portfolios
    ]


@router.post("/portfolios", response_model=PortfolioResponse, status_code=201)
def create_portfolio(
    request: PortfolioCreateRequest,
    db: Session = Depends(get_db),
) -> PortfolioResponse:
    """Create and persist a new portfolio definition."""
    existing = db.query(Portfolio).filter_by(name=request.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Portfolio '{request.name}' already exists.")

    portfolio = Portfolio(
        name=request.name,
        description=request.description,
        asset_tickers=request.asset_tickers,
        initial_capital=request.initial_capital,
    )
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)

    return PortfolioResponse(
        id=portfolio.id,
        name=portfolio.name,
        description=portfolio.description,
        asset_tickers=portfolio.asset_tickers,
        initial_capital=portfolio.initial_capital,
        created_at=portfolio.created_at.isoformat() if portfolio.created_at else None,
        updated_at=portfolio.updated_at.isoformat() if portfolio.updated_at else None,
    )


@router.get("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
def get_portfolio(portfolio_id: int, db: Session = Depends(get_db)) -> PortfolioResponse:
    """Retrieve details for a specific portfolio."""
    portfolio = db.query(Portfolio).filter_by(id=portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found.")

    return PortfolioResponse(
        id=portfolio.id,
        name=portfolio.name,
        description=portfolio.description,
        asset_tickers=portfolio.asset_tickers,
        initial_capital=portfolio.initial_capital,
        created_at=portfolio.created_at.isoformat() if portfolio.created_at else None,
        updated_at=portfolio.updated_at.isoformat() if portfolio.updated_at else None,
    )


@router.get("/portfolios/{portfolio_id}/optimizations")
def get_portfolio_optimizations(
    portfolio_id: int,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Retrieve historical optimization runs for a portfolio."""
    records = db.query(OptimizationRecord).filter_by(portfolio_id=portfolio_id).all()
    return [r.to_dict() for r in records]


@router.get("/portfolios/{portfolio_id}/backtests")
def get_portfolio_backtests(
    portfolio_id: int,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Retrieve historical backtest records for a portfolio."""
    records = db.query(BacktestRecord).filter_by(portfolio_id=portfolio_id).all()
    return [r.to_dict() for r in records]


@router.post("/db/seed", response_model=SeedDatabaseResponse)
def trigger_seed_database(db: Session = Depends(get_db)) -> SeedDatabaseResponse:
    """Manually trigger database seeding with sample market data and sentiment records."""
    try:
        count = seed_database(session=db)
        return SeedDatabaseResponse(
            status="success",
            message="Database successfully seeded with market prices and default portfolio.",
            prices_seeded=count,
        )
    except Exception as exc:
        logger.error("Database seeding failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database seeding failed: {str(exc)}")
