"""
chart_app.py
=================================================================
Renders a single interactive candlestick chart window for the human
trader to label. main.py spawns SIX of these as separate processes
per batch (one per lookback timeframe: 300/200/100/50/20/10 candles),
each on its own local port (8051-8056).

Each window:
  - Shows a candlestick chart for its assigned lookback window.
  - Shows a horizontal volume profile on the right edge of the chart
    (see compute_volume_profile below) so the trader can see where
    volume concentrated across price levels, not just over time.
  - Lets the trader click on the chart to draw a horizontal reference
    line at that price (client-side only, not saved).
  - Shows Buy / Neutral / Sell buttons. Clicking one writes the
    decision into the shared `feedback_dict` (keyed by frame size, e.g.
    300, 200, ...) and closes the browser tab.

This file is not meant to be run directly - show_candles() is called
by main.py via multiprocessing.Process(), once per lookback window.
=================================================================
"""

import threading
import webbrowser
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import os
import sys


def compute_volume_profile(df, num_bins=30, volume_col="volume"):
    """
    Builds a horizontal volume profile: for each candle, its volume is
    distributed proportionally across the price bins its high/low range
    overlaps (instead of just dumping it at the close price).
    Returns (bin_centers, volumes) or (None, None) if no volume column exists.
    """
    if volume_col not in df.columns or df.empty:
        return None, None

    price_min = df["low"].min()
    price_max = df["high"].max()

    if price_min == price_max:
        return None, None

    bins = np.linspace(price_min, price_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    volumes = np.zeros(num_bins)

    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    vols = df[volume_col].to_numpy()

    for low, high, vol in zip(lows, highs, vols):
        if vol == 0:
            continue
        if high <= low:
            # zero-range candle -> dump volume in the single bin it falls in
            idx = np.searchsorted(bins, low) - 1
            idx = min(max(idx, 0), num_bins - 1)
            volumes[idx] += vol
            continue

        overlap_start = np.maximum(bins[:-1], low)
        overlap_end = np.minimum(bins[1:], high)
        overlap = np.clip(overlap_end - overlap_start, 0, None)
        total_overlap = overlap.sum()
        if total_overlap > 0:
            volumes += vol * (overlap / total_overlap)

    return bin_centers, volumes


def show_candles(df, start_candle, end_candle, port, feedback_dict):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Filter the dataframe based on the provided range
    filtered_df = df[
        (df["candle_number"] >= start_candle) & (df["candle_number"] <= end_candle)
        ].copy()

    if filtered_df.empty:
        filtered_df = df

    # Generate initial figure state
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=filtered_df["datetime"],
                open=filtered_df["open"],
                high=filtered_df["high"],
                low=filtered_df["low"],
                close=filtered_df["close"],
                name="Candles",
                increasing_line_color="#089981",
                decreasing_line_color="#F23645",
            )
        ]
    )

    # --- HORIZONTAL VOLUME PROFILE (additive only, no flow changes) ---
    bin_centers, volumes = compute_volume_profile(filtered_df)
    if bin_centers is not None:
        max_vol = volumes.max()
        if max_vol > 0:
            fig.add_trace(
                go.Bar(
                    x=volumes,
                    y=bin_centers,
                    orientation="h",
                    marker_color="rgba(41, 98, 255, 0.35)",
                    xaxis="x2",
                    hoverinfo="skip",
                    showlegend=False,
                    name="Volume Profile",
                )
            )
            fig.update_layout(
                xaxis2=dict(
                    overlaying="x",
                    side="top",
                    range=[max_vol * 4, 0],  # reversed: bars hang from the right edge, grow inward
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    fixedrange=True,
                )
            )

    frame_size = end_candle - start_candle
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111722",
        plot_bgcolor="#111722",
        xaxis=dict(rangeslider=dict(visible=False), gridcolor="#2A2E39", title="Time", type="date"),
        yaxis=dict(gridcolor="#2A2E39", title="Price", side="right"),
        margin=dict(l=50, r=50, t=10, b=10),
        hovermode="x unified",
        uirevision="constant",
        dragmode="pan"
    )

    app = dash.Dash(__name__)

    app.layout = html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "padding": "20px",
            "backgroundColor": "#111722",
            "height": "95vh",
            "display": "flex",
            "flexDirection": "column"
        },
        children=[
            html.H2(f"Candlestick Viewer on frame[{frame_size}] with candle[{end_candle}]",
                    style={"color": "#D1D4DC", "textAlign": "center", "margin": "0"}),

            dcc.Store(id="horizontal-lines-store", data=[]),

            html.Div(
                style={"flex": "1", "minHeight": "60vh"},
                children=[
                    dcc.Graph(
                        id="candlestick-chart",
                        figure=fig,
                        style={"height": "100%", "width": "100%"},
                        config={
                            "scrollZoom": True,
                            "displaylogo": False,
                            "forceGl": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d", "zoom2d", "zoomIn2d", "zoomOut2d"]
                        },
                    )
                ]
            ),

            html.Div(
                style={"textAlign": "center", "padding": "20px 0", "backgroundColor": "#111722"},
                children=[
                    html.Button(
                        "Buy",
                        id="verify-trade-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#089981",
                            "color": "white",
                            "border": "none",
                            "padding": "15px 40px",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "marginRight": "15px",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.4)"
                        }
                    ),

                    html.Button(
                        "Neutral",
                        id="neutral-trade-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#4F5966",
                            "color": "white",
                            "border": "none",
                            "padding": "15px 40px",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "marginRight": "15px",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.4)"
                        }
                    ),

                    html.Button(
                        "Sell",
                        id="reject-trade-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#F23645",
                            "color": "white",
                            "border": "none",
                            "padding": "15px 40px",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.4)"
                        }
                    ),

                    html.Div(id="close-trigger-signal", style={"display": "none"})
                ]
            )
        ],
    )

    # 1. Clientside Line Plot Handler
    app.clientside_callback(
        """
        function(customClickData, existingLines, currentFig) {
            let newLines = existingLines ? [...existingLines] : [];
            let updatedFig = JSON.parse(JSON.stringify(currentFig));
            const trigger = dash_clientside.callback_context.triggered[0];

            if (trigger && trigger.prop_id.includes("candlestick-chart.clickData") && customClickData) {
                let clickedPrice = null;
                if (customClickData.points && customClickData.points[0]) {
                    let p = customClickData.points[0];
                    clickedPrice = (customClickData.event && customClickData.event.yaxis) ? 
                                   customClickData.event.yaxis.d2c(customClickData.event.yval) : p.y;
                }
                if (clickedPrice !== null && !isNaN(clickedPrice)) {
                    if (!newLines.includes(clickedPrice)) {
                        newLines.push(clickedPrice);
                        if (!updatedFig.layout.shapes) updatedFig.layout.shapes = [];
                        updatedFig.layout.shapes.push({
                            type: 'line', xref: 'paper', x0: 0, x1: 1, y0: clickedPrice, y1: clickedPrice,
                            line: { color: '#2962FF', width: 1.5, dash: 'dash' }
                        });
                    }
                }
            }
            return [updatedFig, newLines];
        }
        """,
        Output("candlestick-chart", "figure"),
        Output("horizontal-lines-store", "data"),
        Input("candlestick-chart", "clickData"),
        State("horizontal-lines-store", "data"),
        State("candlestick-chart", "figure"),
        prevent_initial_call=True,
    )

    # 2. Python Callback Handler
    @app.callback(
        Output("close-trigger-signal", "children"),
        Input("verify-trade-btn", "n_clicks"),
        Input("neutral-trade-btn", "n_clicks"),
        Input("reject-trade-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def handle_trade_actions(green_clicks, neutral_clicks, red_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return ""

        triggered_button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_button_id == "verify-trade-btn":
            feedback_dict[frame_size] = "buy"
            print(f"Frame {frame_size}: Accepted")
            return "CLOSE_TAB"

        elif triggered_button_id == "neutral-trade-btn":
            # Saves as False to match default csv states, but prints explicitly
            feedback_dict[frame_size] = "neutral"
            print(f"Frame {frame_size}: Neutral (Skipped)")
            return "CLOSE_TAB"

        elif triggered_button_id == "reject-trade-btn":
            feedback_dict[frame_size] = "sell"
            print(f"Frame {frame_size}: Rejected")
            return "CLOSE_TAB"

        return ""

    # 3. Clientside Close Window Handler
    app.clientside_callback(
        """
        function(signalText) {
            if (signalText === "CLOSE_TAB") {
                window.close();
            }
            return "";
        }
        """,
        Output("verify-trade-btn", "disabled"),
        Input("close-trigger-signal", "children"),
        prevent_initial_call=True
    )

    # 4. PROGRAMMATIC INTERNAL SHUTDOWN ROUTE
    @app.server.route('/shutdown')
    def shutdown_server():
        """Forcefully releases the network socket port associated with this thread context"""
        from flask import request
        func = request.environ.get('werkzeug.server.shutdown')
        if func is not None:
            func()
            return 'Server shutting down...'
        return 'Shutdown requested.'

    def open_browser():
        webbrowser.open_new(f"http://127.0.0.1:{port}/")

    stagger_delay = 0.2 * (port - 8050)
    threading.Timer(1.0 + stagger_delay, open_browser).start()

    app.run(debug=False, port=port, use_reloader=False, threaded=True)