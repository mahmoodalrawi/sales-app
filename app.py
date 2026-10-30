import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="محلل مبيعات المنظفات", layout="wide")
st.markdown("<style>[data-testid=\"stMain\"]{direction:rtl;text-align:right}</style>", unsafe_allow_html=True)

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


GUESS = {
    "date": ["date", "التاريخ", "تاريخ"],
    "governorate": ["governorate", "state", "province", "city", "المحافظة", "محافظة", "المنطقة", "المدينة"],
    "product": ["product", "item", "المنتج", "منتج", "الصنف", "صنف", "المادة"],
    "quantity": ["quantity", "qty", "الكمية", "كمية"],
    "price": ["price", "السعر", "سعر"],
    "total": ["total", "subtotal", "amount", "revenue", "الإجمالي", "الاجمالي", "المجموع", "المبلغ"],
}


def guess(cols, key):
    for c in cols:
        if any(k in str(c).lower() for k in GUESS[key]):
            return str(c)
    return None


def read_raw(file, header):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file, header=header)
    else:
        df = pd.read_excel(file, header=header)
    df.columns = [str(c) for c in df.columns]
    return df


def seasonality(df, by="product"):
    ym = df.groupby([by, df["date"].dt.to_period("M")])["quantity"].sum().reset_index()
    ym["month"] = ym["date"].dt.month
    avg = ym.groupby([by, "month"])["quantity"].mean().unstack()
    idx = avg.div(avg.mean(axis=1), axis=0)
    idx.columns = [MONTHS[m - 1] for m in idx.columns]
    return idx


def season_idx(d):
    m = d.groupby(d["date"].dt.to_period("M"))["quantity"].sum().reset_index()
    m["mo"] = m["date"].dt.month
    a = m.groupby("mo")["quantity"].mean().reindex(range(1, 13))
    a = a.fillna(a.mean())
    return (a / a.mean()).values


def simulate(base, season, start, p, reps, price_pct, promo_pct, runs=40, seed=1):
    """محاكاة مخزون وطلب لمدة 12 شهرًا (مخزون + طلب عشوائي + تأخر التوريد)."""
    rng = np.random.default_rng(seed)
    L, cover = int(p["lead"]), p["cover"]
    cov = (reps / p["reps_now"]) ** 0.5            # أثر زيادة المندوبين (عوائد متناقصة)
    pf = (1 + price_pct / 100) ** p["elast"]        # أثر تغيير السعر على الطلب
    price = p["price0"] * (1 + price_pct / 100)
    cost = p["price0"] * p["cost_pct"] / 100
    units = np.zeros((runs, 12)); profit = np.zeros((runs, 12)); lost = np.zeros(runs)
    for r in range(runs):
        stock = cover * base * season[start % 12]
        pipe = [0.0] * L
        for t in range(12):
            m = (start + t) % 12
            if L:
                stock += pipe.pop(0)
            promo = 1 + promo_pct / 100 if season[m] >= 1.2 else 1.0
            d = base * season[m] * cov * pf * promo * rng.lognormal(0, p["vol"])
            sold = min(d, stock)
            stock -= sold
            lost[r] += d - sold
            fc = base * season[(m + L) % 12] * cov * pf
            order = max(0.0, cover * fc - stock - sum(pipe))
            if L:
                pipe.append(order)
            else:
                stock += order
            units[r, t] = sold
            promo_cost = (promo_pct / 100) * 0.3 * sold * price if promo > 1 else 0
            profit[r, t] = (sold * (price - cost) - stock * cost * p["hold"] / 100
                            - promo_cost - reps * p["rep_cost"])
    return units, profit, lost


COORD = {"بغداد": (33.3152, 44.3661), "نينوى": (36.3350, 43.1189), "البصرة": (30.5085, 47.7804),
         "ذي قار": (31.0439, 46.2576), "بابل": (32.4637, 44.4196), "السليمانية": (35.5558, 45.4351),
         "أربيل": (36.1911, 44.0092), "الأنبار": (33.4258, 43.2993), "النجف": (32.0000, 44.3362),
         "ديالى": (33.7500, 44.6333), "كركوك": (35.4681, 44.3922), "صلاح الدين": (34.6071, 43.6782),
         "واسط": (32.5128, 45.8182), "دهوك": (36.8679, 42.9885), "القادسية": (31.9889, 44.9249),
         "كربلاء": (32.6160, 44.0249), "ميسان": (31.8350, 47.1440), "المثنى": (31.3090, 45.2806)}


def km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (*a, *b))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def fleet_plan(t, hub, p):
    """يحسب المندوبين والسيارات لكل محافظة. t: governorate, pop_m, ppi"""
    t = t.copy()
    t["shops"] = t["pop_m"] * 1000 * p["shops_k"]
    visits = t["shops"] * p["cover"] / 100 * p["visits"]
    t["reps"] = np.maximum(np.ceil(visits / (p["per_day"] * p["days"])), 1).astype(int)
    t["volume"] = p["target_pc"] * t["pop_m"] * 1e6 * t["ppi"]
    t["dist_km"] = [km(COORD[hub], COORD[g]) * 1.3 for g in t["governorate"]]
    days_trip = np.ceil(1 + 2 * t["dist_km"] / p["km_day"])
    trips = p["days"] / days_trip
    t["vehicles"] = np.maximum(np.ceil(t["volume"] / (p["cap"] * trips)), 1).astype(int)
    t["cost"] = t["reps"] * p["rep_cost"] + t["vehicles"] * p["veh_cost"]
    t["cost_per_unit"] = t["cost"] / t["volume"].replace(0, np.nan)
    return t


st.title("محلل مبيعات المنظفات")

with st.expander("إعدادات البيانات والتصفية", expanded=True):
    up = st.file_uploader("ارفع ملف المبيعات (CSV أو Excel)", type=["csv", "xlsx"])
    if up:
        hdr = st.number_input("رقم صف العناوين في الملف", 1, 30, 1)
        raw = read_raw(up, int(hdr) - 1)
        st.caption("أول صفوف الملف:")
        st.dataframe(raw.head(3), use_container_width=True)
        cols = ["—"] + list(raw.columns)

        def pick(label, key):
            g = guess(raw.columns, key)
            return st.selectbox(label, cols, index=cols.index(g) if g else 0)

        c_date = pick("عمود التاريخ", "date")
        c_gov = pick("عمود المحافظة", "governorate")
        c_prod = pick("عمود الصنف", "product")
        c_qty = pick("عمود الكمية", "quantity")
        c_total = pick("عمود إجمالي المبلغ (إن وجد)", "total")
        c_price = pick("عمود سعر الوحدة (إن لم يوجد الإجمالي)", "price")
        if "—" in (c_date, c_gov, c_prod, c_qty) or (c_total == "—" and c_price == "—"):
            st.warning("اختر الأعمدة المطلوبة: التاريخ، المحافظة، الصنف، الكمية، والمبلغ أو السعر.")
            st.stop()
        qty = pd.to_numeric(raw[c_qty], errors="coerce")
        rev = pd.to_numeric(raw[c_total], errors="coerce") if c_total != "—" else qty * pd.to_numeric(raw[c_price], errors="coerce")
        df = pd.DataFrame({
            "date": pd.to_datetime(raw[c_date], errors="coerce"),
            "governorate": raw[c_gov].astype(str).str.strip(),
            "product": raw[c_prod].astype(str).str.strip(),
            "quantity": qty,
            "revenue": rev,
        }).dropna()
        if df.empty:
            st.error("لا توجد صفوف صالحة، تأكد من رقم صف العناوين ومن الأعمدة المختارة.")
            st.stop()
        st.success(f"تم قراءة {len(df):,} صفًا من بياناتك")
    else:
        df = make_sample()
        df["revenue"] = df["quantity"] * df["price"]
        st.info("تعرض الآن بيانات تجريبية")
    gov_filter = st.selectbox("المحافظة", ["كل العراق"] + sorted(df["governorate"].unique()))

view = df if gov_filter == "كل العراق" else df[df["governorate"] == gov_filter]

c1, c2, c3 = st.columns(3)
c1.metric("إجمالي المبيعات", f"{view['revenue'].sum():,.0f}")
c2.metric("عدد الأصناف", view["product"].nunique())
c3.metric("عدد المحافظات", view["governorate"].nunique())

tab1, tab2, tab3, tab4 = st.tabs(["تأثير المواسم على الأصناف", "تنشيط مبيعات المحافظات", "محاكاة السيناريوهات", "الخريطة والتوزيع"])

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

with tab3:
    st.caption("محاكاة 12 شهرًا بمخزون وطلب عشوائي وتأخر توريد، وتقارن وضعك الحالي بسيناريو جديد. "
               "تعتمد على موسمية بياناتك ومتوسط مبيعاتها للمحافظة المختارة.")
    season = season_idx(view)
    n_months = max(view["date"].dt.to_period("M").nunique(), 1)
    base = view["quantity"].sum() / n_months
    price0 = view["revenue"].sum() / max(view["quantity"].sum(), 1)
    start = st.selectbox("شهر البداية", range(12), index=int(view["date"].max().month) % 12,
                         format_func=lambda i: MONTHS[i])
    st.markdown("**الوضع الحالي**")
    reps_now = st.number_input("عدد المندوبين الحالي", 1, 500, 20)
    st.markdown("**السيناريو الجديد**")
    reps = st.slider("عدد المندوبين", 1, 500, int(reps_now))
    price_pct = st.slider("تغيير السعر %", -20, 20, 0)
    promo_pct = st.slider("عرض ترويجي في مواسم الذروة (زيادة طلب %)", 0, 50, 0)
    with st.expander("إعدادات النموذج"):
        lead = st.slider("مدة التوريد (أشهر)", 0, 3, 1)
        cover = st.slider("مخزون الأمان (أشهر تغطية)", 0.5, 3.0, 1.0, 0.1)
        elast = st.slider("مرونة الطلب للسعر", -3.0, -0.2, -1.2, 0.1)
        cost_pct = st.slider("تكلفة المنتج % من السعر", 30, 95, 70)
        vol = st.slider("تذبذب الطلب", 0.0, 0.5, 0.10, 0.05)
        hold = st.slider("تكلفة التخزين الشهرية % من قيمة المخزون", 0.0, 10.0, 2.0, 0.5)
        rep_cost = st.number_input("كلفة المندوب الشهرية (دينار)", 0, 10_000_000, 1_000_000, 100_000)
    p = dict(lead=lead, cover=cover, reps_now=reps_now, elast=elast, price0=price0,
             cost_pct=cost_pct, vol=vol, hold=hold, rep_cost=rep_cost)
    ua, pa, la = simulate(base, season, start, p, reps_now, 0, 0)
    ub, pb, lb = simulate(base, season, start, p, reps, price_pct, promo_pct)

    k1, k2, k3 = st.columns(3)
    k1.metric("الربح السنوي (السيناريو)", f"{pb.sum(1).mean():,.0f}",
              f"{pb.sum(1).mean() - pa.sum(1).mean():,.0f}")
    k2.metric("الوحدات المباعة", f"{ub.sum(1).mean():,.0f}",
              f"{ub.sum(1).mean() - ua.sum(1).mean():,.0f}")
    k3.metric("طلب ضائع بسبب النفاد", f"{lb.mean():,.0f}", f"{lb.mean() - la.mean():,.0f}", delta_color="inverse")

    labels = [MONTHS[(start + t) % 12] for t in range(12)]
    fig = go.Figure()
    for name, u, col in (("الحالي", ua, "#888"), ("السيناريو", ub, "#e8553d")):
        lo, mid, hi = np.percentile(u, [10, 50, 90], axis=0)
        fig.add_trace(go.Scatter(x=labels + labels[::-1], y=list(hi) + list(lo[::-1]), fill="toself",
                                 fillcolor=col, opacity=0.2, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=labels, y=mid, name=name, line=dict(color=col, width=3)))
    fig.update_layout(title="المبيعات الشهرية المتوقعة (الوحدات)", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.caption("خريطة المحافظات مع السكان، وحساب المندوبين والسيارات المطلوبة. القيم الافتراضية افتراضات "
               "يجب استبدالها بأرقامك الفعلية.")
    hub = st.selectbox("موقع المخزن الرئيسي", list(COORD), index=0)
    with st.expander("افتراضات التوزيع", expanded=False):
        shops_k = st.slider("عدد المتاجر لكل 1000 نسمة", 0.5, 8.0, 3.0, 0.5)
        cover = st.slider("نسبة تغطية المتاجر المستهدفة %", 10, 100, 60)
        visits = st.slider("زيارات المتجر الواحد في الشهر", 1, 8, 2)
        per_day = st.slider("زيارات المندوب في اليوم", 5, 40, 15)
        days = st.slider("أيام العمل في الشهر", 18, 28, 24)
        cap = st.number_input("حمولة السيارة (وحدات في الرحلة)", 100, 20000, 2000, 100)
        km_day = st.number_input("مسافة السيارة في اليوم (كم)", 100, 800, 400, 50)
        rep_cost = st.number_input("كلفة المندوب الشهرية (دينار)", 0, 10_000_000, 1_000_000, 100_000, key="rc4")
        veh_cost = st.number_input("كلفة السيارة الشهرية (دينار)", 0, 20_000_000, 3_000_000, 250_000)

    n_m = max(df["date"].dt.to_period("M").nunique(), 1)
    gu = df.groupby("governorate")["quantity"].sum() / n_m
    pops = pd.Series(POP)
    pc = (gu / pops.reindex(gu.index)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pc):
        pc_default = float(pc.quantile(0.75))
    else:
        pc_default = float(gu.sum() / (sum(POP.values()) * 1e6))
    target_pc = st.number_input("هدف المبيعات للفرد شهريًا (وحدات)", 0.0, 1000.0, round(pc_default, 3), 0.01,
                                help="الافتراضي: مبيعات الفرد في أفضل 25% من محافظاتك")

    st.markdown("**السكان والقدرة الشرائية** (1.0 = المتوسط، أعلى = قدرة شرائية أكبر، عدّلها من مسح الدخل الرسمي)")
    t0 = pd.DataFrame({"governorate": list(COORD)})
    t0["pop_m"] = t0["governorate"].map(POP)
    t0["ppi"] = 1.0
    t0 = st.data_editor(t0, disabled=["governorate"], hide_index=True, use_container_width=True, key="fleet_ed")
    p4 = dict(shops_k=shops_k, cover=cover, visits=visits, per_day=per_day, days=days, target_pc=target_pc,
              km_day=km_day, cap=cap, rep_cost=rep_cost, veh_cost=veh_cost)
    plan = fleet_plan(t0, hub, p4)

    m1, m2, m3 = st.columns(3)
    m1.metric("إجمالي المندوبين", f"{plan['reps'].sum():,}")
    m2.metric("إجمالي السيارات", f"{plan['vehicles'].sum():,}")
    m3.metric("الكلفة الشهرية", f"{plan['cost'].sum():,.0f}")

    plan["lat"] = plan["governorate"].map(lambda g: COORD[g][0])
    plan["lon"] = plan["governorate"].map(lambda g: COORD[g][1])
    kw = dict(lat="lat", lon="lon", size="pop_m", color="ppi", hover_name="governorate",
              hover_data={"pop_m": True, "reps": True, "vehicles": True, "lat": False, "lon": False},
              size_max=45, zoom=4.8, height=480, color_continuous_scale="YlOrRd")
    if hasattr(px, "scatter_map"):
        fmap = px.scatter_map(plan, map_style="open-street-map", **kw)
    else:
        fmap = px.scatter_mapbox(plan, mapbox_style="open-street-map", **kw)
    fmap.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fmap, use_container_width=True)

    st.subheader("خطة التوزيع لكل محافظة")
    show = plan[["governorate", "pop_m", "ppi", "shops", "reps", "volume", "dist_km", "vehicles", "cost_per_unit"]]
    st.dataframe(show.sort_values("volume", ascending=False).round(2), hide_index=True, use_container_width=True)
