"""Streamlit dashboard for FORGE frontier visualization.

Run with: streamlit run src/anvil/dashboard/app.py -- /path/to/anvil
"""

from __future__ import annotations

import sys
from pathlib import Path

from anvil.dashboard.data import (
    all_round_points,
    load_frontier,
    load_round_history,
    pareto_frontier_points,
)


def _name(objective: str | dict) -> str:
    return objective if isinstance(objective, str) else objective["name"]


def main() -> None:
    """Render the dashboard, importing optional dependencies only on use."""
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.title("FORGE Frontier Dashboard")
    default = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    repo_root = Path(st.sidebar.text_input("Repository root", str(default)))
    frontier = load_frontier(repo_root)
    if frontier is None:
        st.warning("No frontier found. Run at least one round first.")
        return
    rounds = load_round_history(repo_root)
    directions = frontier.get("directions", {})
    objectives = [
        {"name": objective, "source": objective, "direction": directions.get(objective, "maximize")}
        if isinstance(objective, str)
        else objective
        for objective in frontier.get("objectives", [])
    ]
    points = all_round_points(rounds, objectives)

    st.subheader("Pareto Frontier")
    if len(objectives) >= 2 and points:
        x, y = _name(objectives[0]), _name(objectives[1])
        figure = px.scatter(
            points,
            x=x,
            y=y,
            color="on_frontier",
            hover_data=["round_id"],
            title=f"Pareto Frontier ({x} vs {y})",
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("At least two objectives and one round are needed for the scatter plot.")

    st.subheader("Score Timeline")
    timeline = [
        {
            "round_id": row.get("round_id"),
            "aggregate_score": row.get("aggregate", row.get("aggregate_score")),
        }
        for row in rounds
    ]
    timeline = [row for row in timeline if row["aggregate_score"] is not None]
    if timeline:
        figure = px.line(
            pd.DataFrame(timeline),
            x="round_id",
            y="aggregate_score",
            markers=True,
            title="Aggregate Score by Round",
        )
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Pareto-Optimal Rounds")
    optimal = [point for point in points if point["on_frontier"]]
    st.dataframe(
        pd.DataFrame(optimal or pareto_frontier_points(frontier)), use_container_width=True
    )
    st.subheader("Best So Far")
    progression = [
        {"round_id": row.get("round_id"), **(row.get("frontier_best") or {})}
        for row in rounds
        if row.get("frontier_best")
    ]
    if progression:
        st.dataframe(pd.DataFrame(progression), use_container_width=True)
    st.json(frontier.get("best", {}))


if __name__ == "__main__":
    main()
