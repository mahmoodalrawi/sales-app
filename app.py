import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="محلل مبيعات المنظفات", layout="wide")
st.markdown("<style>.stApp{direction:rtl;text-align:right}</style>", unsafe_allow_html=True)

MONTHS = ["كانون الثاني", "شباط", "آذار", "نيسان", "أيار", "حزيران", "تموز",
          "آب", "أيلول", "تشرين الأول", "تشرين الثاني", "كانون الأول"]

# عدد السكان بالمليون (تقديرات تقريبية، يمكن تعديلها داخل التطبيق)
POP = {"بغداد": 9.0, "نينوى": 3.9, "البصرة": 3.0, "ذي قار": 2.2, "بابل": 2.2,
       "السليمانية": 2.1, "أربيل": 2.0, "الأنبار": 1.9, "النجف": 1.7, "ديالى": 1.7,
       "كركوك": 1.6, "صلاح الدين": 1.6, "واسط": 1.5, "دهوك": 1.4, "القادسية": 1.3,
       "كربلاء": 1.2, "ميسان": 1.1, "المثنى": 0.8}


@st.cache_data
def make_sample():
    """بيانات تجريبية: صنف × محافظة × شهر لثلاث سنوات."""
    rng = np.random.default_rng(7)
    ramadan = {2023: (3, 4), 2024: (3, 4), 2025: (3,)}
    arbaeen = {2023: 9, 2024: 8, 2025: 8}
    # الاسم: (السعر، الكمية الأساسية لكل مليون نسمة، صيف، رمضان، أربعين، ربيع)
    prods = {
        "سائل جلي": (1500, 500, 1.0, 1.5, 1.2, 1.0),
        "مسحوق غسيل": (4000, 400, 0.9, 1.4, 1.1, 1.0),
        "منظف أرضيات": (2000, 450, 1.3, 1.5, 1.4, 1.1),
        "معقم": (2500, 300, 1.5, 1.2, 2.2, 1.0),
        "مبيض": (1200, 350, 1.4, 1.3, 1.2, 1.0),
        "منظف زجاج": (1800, 200, 1.0, 1.2, 1.0, 1.6),
        "صابون سائل": (1700, 350, 1.1, 1.1, 1.8, 1.0),
        "مناديل": (1000, 600, 1.1, 1.2, 2.0, 1.0),
    }
    reach = {g: rng.uniform(0.4, 1.2) for g in POP}  # قوة حضورك في كل محافظة
    rows = []
    for y in (2023, 2024, 2025):
        for m in range(1, 13):
            for g, pop in POP.items():
                for p, (price, base, summer, ram, arb, spr) in prods.items():
                    f = 1.0
                    if m in (6, 7, 8, 9):
                        f *= summer
                    if m in ramadan[y]:
                        f *= ram
                    if m in (4, 5):
                        f *= spr
                    if m == arbaeen[y]:
                        f *= arb ** (1.5 if g in ("كربلاء", "النجف") else 0.6)
                    q = max(int(rng.normal(base * pop * reach[g] * f, base * pop * 0.05)), 0)
                    rows.append((pd.Timestamp(y, m, 1), g, p, q, price))
    return pd.DataFrame(rows, columns=["date", "governorate", "product", "quantity", "price"])


def load(file):
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df["date"] = pd.to_datetime(df["date"])
    return df


def seasonality(df, by="product"):
    ym = df.groupby([by, df["date"].dt.to_period("M")])["quantity"].sum().reset_index()
    ym["month"] = ym["date"].dt.month
    avg = ym.groupby([by, "month"])["quantity"].mean().unstack()
    idx = avg.div(avg.mean(axis=1), axis=0)
    idx.columns = [MONTHS[m - 1] for m in idx.columns]
    return idx


st.title("محلل مبيعات المنظفات")

with st.sidebar:
    st.header("البيانات")
    up = st.file_uploader("ارفع ملف المبيعات (CSV أو Excel)", type=["csv", "xlsx"])
    st.caption("الأعمدة: date, governorate, product, quantity, price")
    df = load(up) if up else make_sample()
    st.info("تعرض الآن بيانات حقيقية" if up else "تعرض الآن بيانات تجريبية")
    gov_filter = st.selectbox("المحافظة", ["كل العراق"] + sorted(df["governorate"].unique()))

df["revenue"] = df["quantity"] * df["price"]
view = df if gov_filter == "كل العراق" else df[df["governorate"] == gov_filter]

c1, c2, c3 = st.columns(3)
c1.metric("إجمالي المبيعات", f"{view['revenue'].sum():,.0f}")
c2.metric("عدد الأصناف", view["product"].nunique())
c3.metric("عدد المحافظات", view["governorate"].nunique())

tab1, tab2 = st.tabs(["تأثير المواسم على الأصناف", "تنشيط مبيعات المحافظات"])

with tab1:
    st.caption("المؤشر 1.0 = المعدل الطبيعي. أعلى من 1.2 موسم ذروة، وأقل من 0.8 موسم ركود.")
    idx = seasonality(view)
    fig = px.imshow(idx.round(2), text_auto=True, aspect="auto",
                    color_continuous_scale="YlOrRd", color_continuous_midpoint=1.0)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("توصيات المخزون")
    for p, row in idx.iterrows():
        peaks = row[row > 1.2].sort_values(ascending=False)
        if len(peaks):
            st.write(f"**{p}**: ذروة في {'، '.join(peaks.index[:3])} — خزّن قبلها بشهر.")

with tab2:
    st.caption("عدّل أعداد السكان إن لزم. الهدف = مبيعات الفرد في أفضل المحافظات.")
    pct = st.slider("مستوى الهدف (أفضل % من المحافظات)", 50, 95, 75)
    g = df.groupby("governorate")["revenue"].sum().reset_index()
    g["pop_m"] = g["governorate"].map(POP).fillna(1.0)
    g = st.data_editor(g, disabled=["governorate", "revenue"], hide_index=True, use_container_width=True)
    g["per_capita"] = g["revenue"] / g["pop_m"]
    target = g["per_capita"].quantile(pct / 100)
    g["gap"] = (target * g["pop_m"] - g["revenue"]).clip(lower=0)
    g = g.sort_values("gap", ascending=False)

    fig2 = px.bar(g, x="governorate", y="gap", title="الفرصة غير المستغلة لكل محافظة")
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("ترتيب الأولوية")
    st.dataframe(g.round(0), hide_index=True, use_container_width=True)
