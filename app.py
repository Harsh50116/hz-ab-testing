"""Hazel A/B Email Testing — Streamlit demo dashboard.

Run with: streamlit run app.py
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

VARIANT_STRATEGY = {
    "seasonal_discount": "Time-limited seasonal offer — creates urgency via a discount window tied to the season.",
    "urgency": "Scarcity messaging — emphasizes limited stock or a closing window to drive immediate action.",
    "personalized_rec": "Social proof + personalization — highlights ratings and reviews for the exact items in the cart.",
}

VARIANT_LABEL = {
    "seasonal_discount": "Seasonal Discount",
    "urgency": "Urgency",
    "personalized_rec": "Personalized Rec",
}

POOL_LABEL = {
    "evening_workday": "Evening (workday)",
    "weekend_noon": "Weekend (noon)",
    "unknown": "Unknown",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    carts = pd.read_csv(DATA_DIR / "carts.csv")
    events = pd.read_csv(DATA_DIR / "engagement_events.csv")
    assignments = pd.read_csv(DATA_DIR / "ab_assignments.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")

    events["occurred_at"] = pd.to_datetime(events["occurred_at"], utc=True)
    carts["abandoned_at"] = pd.to_datetime(carts["abandoned_at"], utc=True)
    events["hour"] = events["occurred_at"].dt.hour
    events["dow"] = events["occurred_at"].dt.day_name()

    return {
        "customers": customers,
        "carts": carts,
        "events": events,
        "assignments": assignments,
        "products": products,
    }


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_email_card(subject: str, body: str, to_name: str | None = None) -> None:
    """Render an email as a styled inbox preview card."""
    to_line = f"to {to_name}" if to_name else ""
    html = f"""
    <div style="
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 18px 22px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        margin-bottom: 12px;
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #f0f2f4;
        ">
            <div style="
                width: 34px; height: 34px;
                border-radius: 50%;
                background: #2c5f2d;
                color: #fff;
                display: flex; align-items: center; justify-content: center;
                font-weight: 600; font-size: 13px;
            ">H</div>
            <div style="font-size: 12px; color: #657786;">
                <strong style="color:#14171a;">Hazel Apparel</strong> &lt;hello@hazelapparel.com&gt;<br/>
                <span style="font-size: 11px;">{to_line}</span>
            </div>
        </div>
        <div style="font-size: 15px; font-weight: 600; color: #14171a; margin-bottom: 10px;">
            {subject}
        </div>
        <div style="font-size: 13px; color: #2d3748; line-height: 1.55;">
            {body}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_row(label_value_pairs: list[tuple[str, str]]) -> None:
    cols = st.columns(len(label_value_pairs))
    for col, (label, value) in zip(cols, label_value_pairs):
        col.markdown(
            f"""
            <div style="padding:4px 0;">
                <div style="font-size:12px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
                <div style="font-size:20px;font-weight:600;color:#f0f0f0;margin-top:2px;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Page 1 — Customer Explorer
# --------------------------------------------------------------------------- #
def page_customer_explorer(data: dict[str, pd.DataFrame]) -> None:
    customers = data["customers"]
    carts = data["carts"]
    events = data["events"]
    assignments = data["assignments"]

    st.title("Customer Explorer")
    st.caption("Click through any customer to see their cart, engagement pattern, and personalized email.")

    # Random button + dropdown
    col_pick, col_rand = st.columns([4, 1])
    if "selected_customer" not in st.session_state:
        st.session_state.selected_customer = customers.iloc[0]["customer_id"]

    with col_rand:
        st.write("")  # spacer to align with selectbox
        st.write("")
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.selected_customer = random.choice(customers["customer_id"].tolist())

    with col_pick:
        options = customers.apply(lambda r: f"{r['customer_id']} — {r['name']}", axis=1).tolist()
        id_to_option = dict(zip(customers["customer_id"], options))
        option_to_id = {v: k for k, v in id_to_option.items()}
        current_option = id_to_option[st.session_state.selected_customer]
        picked = st.selectbox("Select a customer", options, index=options.index(current_option))
        st.session_state.selected_customer = option_to_id[picked]

    cid = st.session_state.selected_customer
    customer = customers[customers["customer_id"] == cid].iloc[0]
    cust_cart = carts[carts["customer_id"] == cid]
    cust_events = events[events["customer_id"] == cid]
    assignment = assignments[assignments["customer_id"] == cid].iloc[0]

    st.divider()

    # Profile card + pool/variant
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Profile")
        st.markdown(
            f"""
            **{customer['name']}** &nbsp; `{customer['customer_id']}`
            <br/><span style="color:#657786;font-size:13px;">{customer['email']}</span>
            """,
            unsafe_allow_html=True,
        )
        render_metric_row([
            ("Segment", customer["segment"].upper()),
            ("Avg Order", f"${customer['avg_order_value']:.0f}"),
            ("Total Orders", str(customer["total_orders"])),
        ])
        st.markdown(
            f"""
            - **Age range:** {customer['age_range']}
            - **Income:** {customer['income_bracket']}
            - **Location:** {customer['city']}, {customer['state']}
            - **Preferred send time (truth):** `{customer['preferred_send_time']}`
            """
        )

    with right:
        st.subheader("Assignment")
        render_metric_row([
            ("Derived Pool", POOL_LABEL.get(assignment["derived_pool"], assignment["derived_pool"])),
            ("Variant", VARIANT_LABEL.get(assignment["assigned_variant"], assignment["assigned_variant"])),
        ])
        st.caption(VARIANT_STRATEGY.get(assignment["assigned_variant"], ""))

    st.divider()

    # Cart
    st.subheader("Abandoned Cart")
    if len(cust_cart) == 0:
        st.info("No cart data for this customer.")
    else:
        cart_display = cust_cart[["product_name", "category", "price", "quantity"]].rename(
            columns={"product_name": "Product", "category": "Category", "price": "Price", "quantity": "Qty"}
        )
        cart_display["Price"] = cart_display["Price"].map(lambda x: f"${x:.0f}")
        st.dataframe(cart_display, hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Cart total", f"${cust_cart.iloc[0]['cart_total']:.0f}")
        c2.metric("Abandoned at", cust_cart.iloc[0]["abandoned_at"].strftime("%b %d, %Y %H:%M UTC"))

    st.divider()

    # Engagement dot-plot
    st.subheader("Engagement Timing")
    st.caption("Each dot is an open or click event — hour of day (UTC). This is what the pool was derived from.")
    if len(cust_events) == 0:
        st.info("No engagement events for this customer.")
    else:
        jitter = [random.Random(i).uniform(-0.25, 0.25) for i in range(len(cust_events))]
        plot_df = cust_events.copy()
        plot_df["y"] = jitter
        fig = px.scatter(
            plot_df,
            x="hour",
            y="y",
            color="event_type",
            hover_data={"occurred_at": True, "dow": True, "hour": True, "y": False, "event_type": True},
            color_discrete_map={"open": "#4a90e2", "click": "#e67e22"},
        )
        fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color="white")))
        fig.update_xaxes(range=[-0.5, 23.5], dtick=2, title="Hour of day (UTC)")
        fig.update_yaxes(range=[-1, 1], showticklabels=False, title="")
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=40), legend_title="")
        # shade evening workday band for context
        fig.add_vrect(x0=17.5, x1=22.5, fillcolor="#2c5f2d", opacity=0.06, line_width=0,
                      annotation_text="evening workday band", annotation_position="top left",
                      annotation_font_size=10)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Generated email
    st.subheader("Generated Email")
    render_email_card(assignment["email_subject"], assignment["email_body"], to_name=customer["name"])


# --------------------------------------------------------------------------- #
# Page 2 — Segmentation Overview
# --------------------------------------------------------------------------- #
def page_segmentation(data: dict[str, pd.DataFrame]) -> None:
    customers = data["customers"]
    assignments = data["assignments"]

    st.title("Segmentation Overview")
    st.caption("How the 100-customer population breaks down across segments, derived pools, and variants.")

    merged = customers.merge(assignments, on="customer_id")

    # Top-line metrics
    render_metric_row([
        ("Customers", str(len(customers))),
        ("Segments", str(customers["segment"].nunique())),
        ("Timing pools", str(assignments["derived_pool"].nunique())),
        ("Variants", str(assignments["assigned_variant"].nunique())),
    ])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segments")
        seg_counts = customers["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.bar(
            seg_counts, x="segment", y="count", text="count",
            color="segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Derived Timing Pools")
        pool_counts = assignments["derived_pool"].value_counts().reset_index()
        pool_counts.columns = ["pool", "count"]
        pool_counts["pool_label"] = pool_counts["pool"].map(POOL_LABEL)
        fig = px.bar(
            pool_counts, x="pool_label", y="count", text="count",
            color="pool_label",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=20, b=40),
                          xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Segment → Pool Breakdown")
    st.caption("Which customer segments land in which timing pools.")
    cross = merged.groupby(["segment", "derived_pool"]).size().reset_index(name="count")
    cross["pool_label"] = cross["derived_pool"].map(POOL_LABEL)
    fig = px.bar(
        cross, x="segment", y="count", color="pool_label", text="count", barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=40), legend_title="Pool")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Variant Assignment Distribution")
    st.caption("The 33/33/34 split landed across the three variants. Randomized within each pool.")
    col_a, col_b = st.columns(2)

    with col_a:
        var_counts = assignments["assigned_variant"].value_counts().reset_index()
        var_counts.columns = ["variant", "count"]
        var_counts["variant_label"] = var_counts["variant"].map(VARIANT_LABEL)
        fig = px.bar(
            var_counts, x="variant_label", y="count", text="count",
            color="variant_label",
            color_discrete_sequence=["#2c5f2d", "#e67e22", "#4a90e2"],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=340, margin=dict(l=20, r=20, t=20, b=40),
                          xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Variants × Pools**")
        pv = assignments.groupby(["derived_pool", "assigned_variant"]).size().reset_index(name="count")
        pv["pool_label"] = pv["derived_pool"].map(POOL_LABEL)
        pv["variant_label"] = pv["assigned_variant"].map(VARIANT_LABEL)
        fig = px.bar(
            pv, x="pool_label", y="count", color="variant_label", text="count", barmode="stack",
            color_discrete_sequence=["#2c5f2d", "#e67e22", "#4a90e2"],
        )
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=40),
                          legend_title="Variant", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Page 3 — Email Variants Side-by-Side
# --------------------------------------------------------------------------- #
def page_email_variants(data: dict[str, pd.DataFrame]) -> None:
    customers = data["customers"]
    assignments = data["assignments"]

    st.title("Email Variants")
    st.caption("Same three strategies, personalized to each customer's cart.")

    merged = assignments.merge(customers[["customer_id", "name"]], on="customer_id")

    variant_order = ["seasonal_discount", "urgency", "personalized_rec"]

    # Header row: strategy + counts + first sample
    st.subheader("Variant Strategies")
    cols = st.columns(3)
    for col, variant in zip(cols, variant_order):
        with col:
            count = (assignments["assigned_variant"] == variant).sum()
            st.markdown(
                f"""
                <div style="padding: 12px; background: #f7f9fb; border-radius: 6px; margin-bottom: 10px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #657786; letter-spacing: 0.5px;">Variant</div>
                    <div style="font-size: 18px; font-weight: 600; color: #14171a;">{VARIANT_LABEL[variant]}</div>
                    <div style="font-size: 12px; color: #657786; margin-top: 4px;">{count} customers assigned</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(VARIANT_STRATEGY[variant])

            sample = merged[merged["assigned_variant"] == variant].iloc[0]
            render_email_card(sample["email_subject"], sample["email_body"], to_name=sample["name"])

    st.divider()

    # Personalization row — same variant, different customer → different cart → different email
    st.subheader("Personalization in Action")
    st.caption(
        "Same variant, different customer. The strategy is fixed but the content adapts "
        "to each person's cart."
    )
    cols = st.columns(3)
    for col, variant in zip(cols, variant_order):
        with col:
            st.markdown(f"**{VARIANT_LABEL[variant]} — second customer**")
            pool = merged[merged["assigned_variant"] == variant]
            if len(pool) >= 2:
                sample = pool.iloc[1]
                render_email_card(sample["email_subject"], sample["email_body"], to_name=sample["name"])
            else:
                st.info("Not enough samples.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Hazel A/B Email Demo",
        page_icon="📧",
        layout="wide",
    )

    data = load_data()

    st.sidebar.title("Hazel A/B Demo")
    st.sidebar.caption("Email testing pipeline walkthrough")
    page = st.sidebar.radio(
        "Page",
        ["Customer Explorer", "Segmentation Overview", "Email Variants"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
        **Dataset**
        - {len(data['customers'])} customers
        - {len(data['events'])} engagement events
        - {len(data['assignments'])} generated emails
        """
    )

    if page == "Customer Explorer":
        page_customer_explorer(data)
    elif page == "Segmentation Overview":
        page_segmentation(data)
    else:
        page_email_variants(data)


if __name__ == "__main__":
    main()
