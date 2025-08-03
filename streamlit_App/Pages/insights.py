import streamlit as st

def show_insights_page():
    st.title("📌 Strategic Recommendations")

    # Custom CSS for styling recommendation cards
    st.markdown("""
        <style>
            .rec-card {
                background: #ffffff;
                border-radius: 14px;
                padding: 18px 22px;
                margin-bottom: 14px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border-left: 5px solid #2563eb;
            }
            .rec-card h4 {
                margin: 0;
                font-size: 1.1rem;
                font-weight: 700;
                color: #1f2937;
            }
            .rec-card p {
                margin: 6px 0 0;
                color: #374151;
                font-size: 0.95rem;
                line-height: 1.4;
            }
            .tag {
                display: inline-block;
                background: #e0f2fe;
                color: #0369a1;
                font-size: 0.7rem;
                font-weight: 600;
                padding: 3px 8px;
                border-radius: 999px;
                margin-top: 6px;
            }
        </style>
    """, unsafe_allow_html=True)

    recommendations = [
        {
            "icon": "💳",
            "title": "Tiered Loyalty & Upsell Programs",
            "desc": "Target **Regular** and **Core Shoppers** with reward tiers and bundled offers to increase order value.",
            "tag": "High Impact"
        },
        {
            "icon": "🎯",
            "title": "Premium Segment Engagement",
            "desc": "Run highly personalized campaigns for **Premium Shoppers** to increase purchase frequency and wallet share.",
            "tag": "Medium Impact"
        },
        {
            "icon": "🏢",
            "title": "B2B Relationship Building",
            "desc": "Develop a cadence program for **Wholesale Buyers**, offering bulk order incentives and contract deals.",
            "tag": "High Impact"
        },
        {
            "icon": "📩",
            "title": "Onboarding & Reactivation",
            "desc": "Send automated welcome flows and reactivation offers to **new cohorts** to reduce early churn.",
            "tag": "High Priority"
        },
        {
            "icon": "📊",
            "title": "Data-Driven Seasonal Planning",
            "desc": "Leverage forecast peaks to time promotions and optimize inventory levels before demand surges.",
            "tag": "Strategic"
        }
    ]

    for rec in recommendations:
        st.markdown(
            f"""
            <div class="rec-card">
                <h4>{rec['icon']} {rec['title']}</h4>
                <p>{rec['desc']}</p>
                <div class="tag">{rec['tag']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
