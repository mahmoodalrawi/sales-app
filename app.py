import numpy as np
import requests
import time
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


SHOP_TYPES = {
    "سوبرماركت / ماركت": ["supermarket"],
    "بقالة / محل مواد غذائية": ["convenience", "grocery", "general"],
    "مول / متجر كبير": ["mall", "department_store"],
    "جملة": ["wholesale"],
    "كشك / محل صغير": ["kiosk", "variety_store"],
    "محل مواد تجميل وتنظيف": ["chemist"],
}
TYPE_AR = {"supermarket": "سوبرماركت", "convenience": "بقالة", "grocery": "بقالة", "general": "محل عام",
           "mall": "مول", "department_store": "متجر كبير", "wholesale": "جملة", "kiosk": "كشك",
           "variety_store": "محل صغير", "chemist": "تجميل وتنظيف"}
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]


def parse_osm(js):
    rows = []
    for el in js.get("elements", []):
        c = el if "lat" in el else el.get("center")
        if not c:
            continue
        tg = el.get("tags", {})
        rows.append({"name": tg.get("name:ar") or tg.get("name") or tg.get("name:en") or "بدون اسم",
                     "type": TYPE_AR.get(tg.get("shop", ""), tg.get("shop", "")),
                     "lat": c["lat"], "lon": c["lon"]})
    return pd.DataFrame(rows, columns=["name", "type", "lat", "lon"])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_shops(lat, lon, radius_km, tags):
    q = (f'[out:json][timeout:50];nwr["shop"~"^({"|".join(tags)})$"]'
         f'(around:{int(radius_km * 1000)},{lat},{lon});out center 4000;')
    last = None
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": q}, headers={"User-Agent": "sales-analyzer/1.0"}, timeout=70)
            r.raise_for_status()
            return parse_osm(r.json())
        except Exception as e:
            last = e
    raise RuntimeError(str(last))


def uncovered(shops, mine, radius_km):
    """True للمتجر الذي لا توجد نقطة بيع لك ضمن نصف القطر."""
    if mine.empty or shops.empty:
        return pd.Series(True, index=shops.index)
    la1 = np.radians(shops["lat"].values)[:, None]
    lo1 = np.radians(shops["lon"].values)[:, None]
    la2 = np.radians(mine["lat"].values)[None, :]
    lo2 = np.radians(mine["lon"].values)[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    d = 6371 * 2 * np.arcsin(np.sqrt(h))
    return pd.Series(d.min(axis=1) > radius_km, index=shops.index)


def run_abm(p, shops, hub, season, seed=1):
    """محاكاة وكلاء ساعة بساعة: متاجر + مخزن + شاحنات + مندوبون."""
    rng = np.random.default_rng(seed)
    n, H, D = len(shops), 10, int(p["days"])
    lat0, lon0 = hub
    kx = 111.0 * np.cos(np.radians(lat0))
    xy = np.c_[(shops["lon"].values - lon0) * kx, (shops["lat"].values - lat0) * 111.0]
    lam = rng.lognormal(0, 0.5, n)
    lam = lam / lam.mean() * p["demand"]
    mx, rop = lam * p["max_days"], lam * p["rop_days"]
    stock = mx * 0.8
    pending, assigned = np.zeros(n, bool), np.zeros(n, bool)
    qty = np.zeros(n)
    last_visit = np.full(n, -99.0)
    W = lam.sum() * p["wh_days"]
    T, R = int(p["trucks"]), int(p["reps"])
    tpos = np.zeros((T, 2))
    tstate, troute = ["idle"] * T, [[] for _ in range(T)]
    order = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
    rroutes = [r for r in np.array_split(order, R) if len(r)] if R else []
    sp = p["speed"] / 1.3
    frames, daily, util = [], [], []
    sold_tot = dem_tot = km = 0.0
    deliveries = 0
    for day in range(D):
        sf = season[(p["start"] + day // 30) % 12]
        if day and day % p["restock_every"] == 0:
            W += lam.sum() * p["restock_every"] * p["supply_pct"] / 100
        s_day = d_day = 0.0
        for h in range(H):
            t = day * H + h
            boost = np.where(day - last_visit <= 7, 1 + p["rep_eff"] / 100, 1.0)
            d = lam / H * sf * boost * rng.gamma(4, 0.25, n)
            s = np.minimum(d, stock)
            stock -= s
            s_day += s.sum(); d_day += d.sum()
            need = (stock <= rop) & ~pending
            qty[need] = mx[need] - stock[need]
            pending |= need
            reps_xy = []
            for rt in rroutes:
                i = rt[t % len(rt)]
                last_visit[i] = day
                reps_xy.append(xy[i])
            for k in range(T):
                if tstate[k] != "idle":
                    continue
                cand = list(np.where(pending & ~assigned)[0])
                if not cand or W <= 1:
                    continue
                pos, load, route = tpos[k], 0.0, []
                while cand and load < p["cap"] and W - load > 1:
                    j = int(np.argmin(np.hypot(*(xy[cand] - pos).T)))
                    i = cand.pop(j)
                    q = min(qty[i], p["cap"] - load, W - load)
                    if q <= 0:
                        break
                    route.append((i, q)); load += q; pos = xy[i]
                if route:
                    for i, q in route:
                        assigned[i] = True
                    W -= load
                    troute[k], tstate[k] = route, "out"
                    util.append(load / p["cap"])
            for k in range(T):
                budget = sp
                while tstate[k] != "idle" and budget > 1e-9:
                    tgt = xy[troute[k][0][0]] if troute[k] else np.zeros(2)
                    vec = tgt - tpos[k]
                    dist = float(np.hypot(*vec))
                    if dist <= budget:
                        tpos[k] = tgt; budget -= dist; km += dist * 1.3
                        if troute[k]:
                            i, q = troute[k].pop(0)
                            stock[i] += q; pending[i] = False; assigned[i] = False
                            deliveries += 1; budget -= sp * 0.25
                        else:
                            tstate[k] = "idle"
                    else:
                        tpos[k] = tpos[k] + vec / dist * budget; km += budget * 1.3; budget = 0
            if t % p["snap"] == 0:
                frames.append(dict(day=day + 1, hour=h + 1, ratio=np.clip(stock / mx, 0, 1).copy(),
                                   trucks=tpos.copy(), reps=np.array(reps_xy).reshape(-1, 2)))
        sold_tot += s_day; dem_tot += d_day
        daily.append(dict(day=day + 1, service=100 * s_day / max(d_day, 1e-9), stockout=int((stock <= 1e-6).sum())))
    cost = T * p["truck_day"] * D + km * p["km_cost"] + R * p["rep_day"] * D
    k = dict(service=100 * sold_tot / max(dem_tot, 1e-9), lost=dem_tot - sold_tot, sold=sold_tot, km=km,
             deliveries=deliveries, util=100 * (np.mean(util) if util else 0), cost=cost,
             profit=sold_tot * p["margin"] - cost, cpu=cost / max(sold_tot, 1e-9))
    rad = float(np.hypot(xy[:, 0], xy[:, 1]).max())
    return dict(frames=frames, daily=pd.DataFrame(daily), kpi=k, geo=(lat0, lon0, kx),
                shops_lat=shops["lat"].values, shops_lon=shops["lon"].values,
                zoom=float(np.clip(11.4 - np.log2(max(rad, 3) / 10), 7, 13)))


def abm_fig(res, i):
    f = res["frames"][i]
    lat0, lon0, kx = res["geo"]
    S = go.Scattermap if hasattr(go, "Scattermap") else go.Scattermapbox
    fig = go.Figure()
    fig.add_trace(S(lat=res["shops_lat"], lon=res["shops_lon"], mode="markers", name="متاجر",
                    marker=dict(size=9, color=f["ratio"], colorscale="RdYlGn", cmin=0, cmax=1)))
    fig.add_trace(S(lat=[lat0], lon=[lon0], mode="markers", name="المخزن", marker=dict(size=18, color="#1f4fff")))
    for key, name, col, size in (("trucks", "شاحنات", "#ff7a00", 14), ("reps", "مندوبون", "#8e44ad", 11)):
        a = f[key]
        if len(a):
            fig.add_trace(S(lat=lat0 + a[:, 1] / 111.0, lon=lon0 + a[:, 0] / kx, mode="markers", name=name,
                            marker=dict(size=size, color=col)))
    layout_key = "map" if hasattr(go, "Scattermap") else "mapbox"
    fig.update_layout(**{layout_key: dict(style="open-street-map", center=dict(lat=lat0, lon=lon0), zoom=res["zoom"])},
                      margin=dict(l=0, r=0, t=30, b=0), height=480, legend=dict(orientation="h", y=1.0),
                      title=f"اليوم {f['day']} - الساعة {f['hour']}")
    return fig


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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["تأثير المواسم على الأصناف", "تنشيط مبيعات المحافظات", "محاكاة السيناريوهات", "الخريطة والتوزيع", "محاكاة التوزيع (وكلاء)"])

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
    pc = (gu / (pops.reindex(gu.index) * 1e6)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pc):
        pc_default = float(pc.quantile(0.75))
    else:
        pc_default = float(gu.sum() / (sum(POP.values()) * 1e6))
    pc_default = float(pc_default) if np.isfinite(pc_default) and pc_default > 0 else 0.001
    target_pc = st.number_input("هدف المبيعات للفرد شهريًا (وحدات)", min_value=0.0, value=round(pc_default, 5),
                                step=0.001, format="%.5f",
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

    st.divider()
    st.subheader("نقاط البيع داخل المحافظة")
    st.caption("تُجلب المحلات والماركتات من خرائط OpenStreetMap المفتوحة. تغطيتها في العراق غير مكتملة، "
               "فالأدق أن ترفع قائمة عملائك الفعلية بإحداثياتهم من Odoo لمقارنتها.")
    g_sel = st.selectbox("اختر المحافظة", list(COORD), index=0, key="g_sel")
    radius = st.slider("نصف قطر البحث حول مركز المحافظة (كم)", 3, 40, 15)
    kinds = st.multiselect("نوع المحل", list(SHOP_TYPES), default=list(SHOP_TYPES)[:3])
    mine_up = st.file_uploader("نقاط بيعك الحالية (اختياري): أعمدة name, lat, lon", type=["csv", "xlsx"], key="mine")
    cov_km = st.slider("نصف قطر تغطية نقطة بيعك (كم)", 0.3, 5.0, 1.0, 0.1)

    if st.button("ابحث عن نقاط البيع"):
        tags = tuple(sorted({t for k in kinds for t in SHOP_TYPES[k]}))
        if not tags:
            st.warning("اختر نوع محل واحدًا على الأقل.")
        else:
            with st.spinner("جارٍ جلب نقاط البيع..."):
                try:
                    st.session_state["shops"] = (g_sel, radius, fetch_shops(*COORD[g_sel], radius, tags))
                except Exception as e:
                    st.session_state.pop("shops", None)
                    st.error("تعذر جلب البيانات الآن من الخادم المجاني، أعد المحاولة بعد دقيقة أو قلّل نصف القطر.")

    if "shops" in st.session_state:
        g_done, rad_done, shops = st.session_state["shops"]
        if shops.empty:
            st.info("لم يُعثر على محلات مسجلة في هذه المنطقة. خرائط OSM ناقصة في بعض المناطق، "
                    "جرّب نصف قطر أكبر أو ارفع قائمة عملائك.")
        else:
            mine = pd.DataFrame(columns=["name", "lat", "lon"])
            if mine_up:
                m = read_raw(mine_up, 0)
                m.columns = [c.lower().strip() for c in m.columns]
                m = m.rename(columns={"latitude": "lat", "longitude": "lon", "lng": "lon"})
                if {"lat", "lon"} <= set(m.columns):
                    m["lat"] = pd.to_numeric(m["lat"], errors="coerce")
                    m["lon"] = pd.to_numeric(m["lon"], errors="coerce")
                    mine = m.dropna(subset=["lat", "lon"])
                    if "name" not in mine.columns:
                        mine["name"] = "نقطتك"
                else:
                    st.warning("الملف يجب أن يحتوي العمودين lat و lon.")
            shops = shops.copy()
            if len(mine):
                shops["الفئة"] = np.where(uncovered(shops, mine, cov_km), "غير مغطى", "مغطى بنقطتك")
                pts = pd.concat([shops, mine.assign(type="نقطتك", الفئة="نقطتك")[["name", "type", "lat", "lon", "الفئة"]]])
            else:
                shops["الفئة"] = shops["type"]
                pts = shops

            c1, c2 = st.columns(2)
            c1.metric("عدد المحلات في " + g_done, f"{len(shops):,}")
            if len(mine):
                c2.metric("محلات غير مغطاة", f"{(shops['الفئة'] == 'غير مغطى').sum():,}")
            else:
                c2.metric("أنواع المحلات", shops["type"].nunique())

            zoom = float(np.clip(11.4 - np.log2(rad_done / 10), 7, 13))
            kw2 = dict(lat="lat", lon="lon", color="الفئة", hover_name="name", hover_data={"type": True, "lat": False, "lon": False},
                       zoom=zoom, height=520, center={"lat": COORD[g_done][0], "lon": COORD[g_done][1]})
            if hasattr(px, "scatter_map"):
                f2 = px.scatter_map(pts, map_style="open-street-map", **kw2)
            else:
                f2 = px.scatter_mapbox(pts, mapbox_style="open-street-map", **kw2)
            f2.update_traces(marker=dict(size=9))
            f2.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(f2, use_container_width=True)

            st.dataframe(shops["type"].value_counts().rename("العدد").reset_index(), hide_index=True,
                         use_container_width=True)
            st.download_button("تحميل القائمة (CSV)", shops.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"points_{g_done}.csv", mime="text/csv")


with tab5:
    st.caption("نموذج وكلاء لبغداد: متاجر ومخزن وشاحنات ومندوبون يعملون ساعة بساعة. الأرقام الافتراضية افتراضات "
               "عدّلها بأرقامك. المتاجر من نتائج بحث بغداد إن وُجدت، وإلا فهي نقاط تجريبية موزعة حول المركز.")
    season5 = season_idx(view)
    price5 = view["revenue"].sum() / max(view["quantity"].sum(), 1)
    with st.expander("إعدادات النموذج", expanded=True):
        n_shops = st.slider("عدد المتاجر", 30, 300, 120, 10)
        trucks = st.slider("عدد الشاحنات", 1, 20, 4)
        cap5 = st.number_input("حمولة الشاحنة (وحدات)", 50, 10000, 800, 50, key="cap5")
        speed = st.slider("سرعة الشاحنة في المدينة (كم/س)", 10, 60, 25)
        reps5 = st.slider("عدد المندوبين", 0, 30, 6, key="reps5")
        rep_eff = st.slider("أثر زيارة المندوب على طلب المتجر %", 0, 40, 10)
        demand5 = st.number_input("متوسط طلب المتجر اليومي (وحدات)", 1.0, 1000.0, 20.0, 1.0)
        days5 = st.slider("مدة المحاكاة (أيام)", 5, 30, 14)
    with st.expander("إعدادات متقدمة"):
        rop_days = st.slider("إعادة الطلب عند مخزون (أيام)", 0.5, 5.0, 2.0, 0.5)
        max_days = st.slider("تعبئة المتجر إلى (أيام)", 2.0, 15.0, 6.0, 0.5)
        wh_days = st.slider("مخزون المخزن الابتدائي (أيام)", 1, 30, 5)
        restock_every = st.slider("توريد المخزن كل (أيام)", 1, 14, 5)
        supply_pct = st.slider("كمية التوريد % من الاستهلاك", 30, 150, 100)
        truck_day = st.number_input("كلفة الشاحنة اليومية (دينار)", 0, 5_000_000, 100_000, 10_000)
        km_cost = st.number_input("كلفة الكيلومتر (دينار)", 0, 5000, 500, 50)
        rep_day = st.number_input("كلفة المندوب اليومية (دينار)", 0, 1_000_000, 40_000, 5_000, key="rd5")
        margin = st.number_input("ربح الوحدة (دينار)", 0, 100_000, max(int(price5 * 0.3), 1), 50)
        hub_lat = st.number_input("خط عرض المخزن", 29.0, 38.0, 33.3152, 0.001, format="%.4f")
        hub_lon = st.number_input("خط طول المخزن", 38.0, 49.0, 44.3661, 0.001, format="%.4f")

    if st.button("شغّل المحاكاة", key="run_abm"):
        src = st.session_state.get("shops")
        if src is not None and src[0] == "بغداد" and len(src[2]) >= 20:
            sh = src[2].sample(min(n_shops, len(src[2])), random_state=1)[["lat", "lon"]].reset_index(drop=True)
            st.session_state["abm_src"] = "نتائج البحث"
        else:
            rg = np.random.default_rng(5)
            sh = pd.DataFrame({"lat": 33.3152 + rg.normal(0, 0.06, n_shops), "lon": 44.3661 + rg.normal(0, 0.07, n_shops)})
            st.session_state["abm_src"] = "نقاط تجريبية"
        pa5 = dict(demand=demand5, days=days5, cap=cap5, speed=speed, rep_eff=rep_eff, rop_days=rop_days,
                   max_days=max_days, wh_days=wh_days, restock_every=restock_every, supply_pct=supply_pct,
                   truck_day=truck_day, km_cost=km_cost, rep_day=rep_day, margin=margin, trucks=trucks, reps=reps5,
                   start=int(view["date"].max().month) % 12, snap=2)
        with st.spinner("جارٍ المحاكاة..."):
            res = run_abm(pa5, sh, (hub_lat, hub_lon), season5)
        st.session_state["abm"] = res
        st.session_state["abm_frame"] = 0
        k = res["kpi"]
        st.session_state.setdefault("abm_hist", []).append({
            "شاحنات": trucks, "مندوبون": reps5, "الحمولة": cap5, "مستوى الخدمة %": round(k["service"], 1),
            "طلب ضائع": round(k["lost"]), "كم": round(k["km"]), "ربح": round(k["profit"]),
            "كلفة/وحدة": round(k["cpu"], 1)})

    if "abm" in st.session_state:
        res = st.session_state["abm"]
        k = res["kpi"]
        st.caption("مصدر المتاجر: " + st.session_state.get("abm_src", ""))
        a1, a2 = st.columns(2)
        a1.metric("مستوى الخدمة", f"{k['service']:.1f}%")
        a2.metric("طلب ضائع (وحدات)", f"{k['lost']:,.0f}")
        a3, a4 = st.columns(2)
        a3.metric("كيلومترات الشاحنات", f"{k['km']:,.0f}")
        a4.metric("الربح بعد الكلف", f"{k['profit']:,.0f}")
        st.caption(f"استغلال حمولة الشاحنات: {k['util']:.0f}% | عدد التسليمات: {k['deliveries']:,} | كلفة الوحدة: {k['cpu']:,.1f}")

        ph = st.empty()
        fr = st.slider("اللحظة", 0, len(res["frames"]) - 1, 0, key="abm_frame")
        if st.button("▶ تشغيل متحرك", key="abm_play"):
            for i in range(fr, len(res["frames"])):
                ph.plotly_chart(abm_fig(res, i), use_container_width=True, key=f"abm_play_{i}")
                time.sleep(0.25)
        else:
            ph.plotly_chart(abm_fig(res, fr), use_container_width=True, key="abm_static")
        st.caption("لون المتجر: أخضر = مخزون جيد، أحمر = قارب على النفاد. برتقالي = شاحنات، بنفسجي = مندوبون، أزرق = المخزن.")
        st.markdown("**مستوى الخدمة اليومي %**")
        st.line_chart(res["daily"].set_index("day")["service"])
        st.markdown("**عدد المتاجر التي نفد مخزونها نهاية اليوم**")
        st.bar_chart(res["daily"].set_index("day")["stockout"])
        st.markdown("**سجل التجارب** (غيّر الإعدادات وأعد التشغيل للمقارنة)")
        st.dataframe(pd.DataFrame(st.session_state["abm_hist"]), hide_index=True, use_container_width=True)
