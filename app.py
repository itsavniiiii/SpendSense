"""SpendSense backend (FastAPI + SQLite + pandas + scikit-learn). All data is synthetic."""
import os, io, csv, re, json, time, sqlite3, hashlib
from decimal import Decimal, InvalidOperation
import numpy as np, pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DB = os.environ.get("SPENDSENSE_DB", "spendsense.db")
HERE = os.path.dirname(os.path.abspath(__file__))
TH = 0.40  # below this ML confidence -> "Other" + needs review
CATS = ["Housing","Groceries","Restaurants","Shopping","Transportation","Travel","Entertainment",
        "Subscriptions","Healthcare","Utilities","Education","Income","Other"]
R = {"whole foods":"Groceries","trader joe":"Groceries","walmart":"Groceries","target":"Shopping",
     "amazon prime":"Subscriptions","amazon":"Shopping","starbucks":"Restaurants","chipotle":"Restaurants",
     "uber":"Transportation","lyft":"Transportation","shell":"Transportation","netflix":"Subscriptions",
     "hulu":"Subscriptions","spotify":"Subscriptions","adobe":"Subscriptions","apple icloud":"Subscriptions",
     "planet fitness":"Subscriptions","delta":"Travel","cvs":"Healthcare","greenfield":"Housing",
     "georgia power":"Utilities","comcast":"Utilities","payroll":"Income","freelance":"Income",
     "amc":"Entertainment","udemy":"Education"}
KEYS = sorted(R, key=len, reverse=True)
FREQ = [("weekly",6,8,52,pd.DateOffset(weeks=1)),("monthly",27,33,12,pd.DateOffset(months=1)),
        ("quarterly",85,95,4,pd.DateOffset(months=3)),("annual",350,380,1,pd.DateOffset(years=1))]
OVERLAP = {"streaming": {"netflix","hulu","disney plus"}}
ACCT = {1: "Checking", 2: "Credit Card"}
METRICS = {}

SCHEMA = """
create table if not exists users(id integer primary key,name text,email text,created_at text);
create table if not exists accounts(id integer primary key,user_id int,account_name text,account_type text,currency text);
create table if not exists import_batches(id integer primary key autoincrement,user_id int,uploaded_at text default current_timestamp,
 records_received int,records_imported int,duplicates_skipped int,records_rejected int);
create table if not exists transactions(id integer primary key autoincrement,user_id int,account_id int,merchant text,norm text,
 description text,amount_cents int not null,currency text default 'USD',transaction_type text,transaction_date text not null,
 category text,category_confidence real,needs_review int default 0,recurring int default 0,import_batch_id int,
 created_at text default current_timestamp,hash text unique);
create index if not exists ix_tx_date on transactions(user_id,transaction_date);
create index if not exists ix_tx_cat on transactions(category);
create table if not exists user_map(norm text primary key,category text);
create table if not exists subscriptions(id integer primary key autoincrement,user_id int,merchant text,kind text,billing_frequency text,
 estimated_amount real,monthly_equivalent real,next_expected_payment text,last_payment text,confidence real,status text,price_change text,n int,history text);
create table if not exists anomalies(id integer primary key autoincrement,transaction_id int,anomaly_score real,explanation text,
 detection_method text,user_feedback text);
create table if not exists forecasts(id integer primary key autoincrement,user_id int,forecast_period text,predicted_amount real,
 lower_bound real,upper_bound real,model_version text,detail text);
create table if not exists recommendations(id integer primary key autoincrement,user_id int,key text,recommendation_type text,title text,
 description text,monthly_savings_estimate real,annual_savings_estimate real,confidence real,status text,action text,evidence text,counted int);
"""

def con():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
def Q(sql, a=(), one=False):
    c = con(); r = [dict(x) for x in c.execute(sql, a).fetchall()]; c.close()
    return (r[0] if r else None) if one else r
def X(sql, a=()):
    c = con(); c.execute(sql, a); c.commit(); c.close()
def INS(t, r):
    X(f"insert into {t}({','.join(r)}) values({','.join('?'*len(r))})", tuple(r.values()))

# ---------------- categorization: rules -> TF-IDF/LogReg -> user corrections ----------------
def norm(m):
    c = re.sub(r"[^a-z ]", "", (m or "").lower().replace("'", "")).strip()
    for k in KEYS:
        if re.search(r"\b" + k, c): return k
    return " ".join(c.split()[:3]) or "unknown"

VOC = {"Groceries":"grocery market produce supermarket organic","Restaurants":"cafe coffee grill steakhouse pizza restaurant bistro kitchen",
 "Shopping":"store shop outlet mall boutique online order","Transportation":"fuel gas rideshare parking transit taxi",
 "Travel":"airline hotel flight resort booking","Entertainment":"cinema theater tickets concert games",
 "Subscriptions":"subscription membership monthly streaming plan","Healthcare":"pharmacy clinic dental medical health",
 "Utilities":"electric water internet power utility","Education":"tuition course school books university",
 "Housing":"rent property apartments lease mortgage","Income":"payroll deposit salary freelance","Other":"misc payment transfer pos"}
def train():
    r = np.random.default_rng(7); X_, y = [], []
    for c, v in VOC.items():
        w = v.split()
        for _ in range(60):
            k = r.integers(1, 4); t = list(r.choice(w, k)) + [f"zz{r.integers(0,40)}", f"qq{r.integers(0,40)}"]
            r.shuffle(t); X_.append(" ".join(t)); y.append(c)
    a, b, ya, yb = train_test_split(X_, y, test_size=.25, random_state=1, stratify=y)
    m = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500)).fit(a, ya)
    METRICS["categorizer_heldout_accuracy"] = round(float((m.predict(b) == np.array(yb)).mean()), 3)
    METRICS["categorizer_note"] = "Trained and tested on synthetic descriptions; optimistic vs. real bank data."
    return m
MODEL = train()

def categorize(n, text, um):
    if n in um: return um[n], 1.0
    for k in KEYS:
        if k == n: return R[k], 0.98
    p = MODEL.predict_proba([text.lower()])[0]; i = int(p.argmax())
    return str(MODEL.classes_[i]), round(float(p[i]), 3)

# ---------------- synthetic data ----------------
def gen():
    r = np.random.default_rng(42); rows = []
    def add(d, m, desc, amt, acct): rows.append((d, m, desc, int(round(amt * 100)), acct))
    pick = lambda xs: xs[int(r.integers(len(xs)))]
    for i, p in enumerate(pd.period_range("2025-01", "2026-10", freq="M")):
        y, mo = p.year, p.month; D = lambda day: f"{y}-{mo:02d}-{int(day):02d}"
        add(D(1), "GREENFIELD PROPERTIES", "RENT PAYMENT", -1850, 1)
        add(D(12), "GEORGIA POWER", "ELECTRIC UTILITY", -round(r.uniform(90, 140), 2), 1)
        add(D(18), "COMCAST XFINITY", "INTERNET UTILITY", -79.99, 1)
        add(D(1), "ACME CORP PAYROLL", "PAYROLL DEPOSIT", 3200, 1); add(D(15), "ACME CORP PAYROLL", "PAYROLL DEPOSIT", 3200, 1)
        if r.random() < .3: add(D(20), "FREELANCE DEPOSIT", "FREELANCE INCOME", round(r.uniform(400, 1500), 2), 1)
        add(D(8), "NETFLIX.COM", "NETFLIX STREAMING", -(15.99 if (y, mo) < (2026, 4) else 22.99), 2)
        add(D(20), "HULU", "HULU STREAMING", -17.99, 2); add(D(5), "SPOTIFY", "SPOTIFY PREMIUM", -10.99, 2)
        add(D(22), "ADOBE CREATIVE CLOUD", "ADOBE SUBSCRIPTION", -(54.99 if y < 2026 else 59.99), 2)
        add(D(3), "PLANET FITNESS", "PLANET FITNESS MEMBERSHIP", -24.99, 2); add(D(10), "APPLE ICLOUD", "APPLE ICLOUD STORAGE", -2.99, 2)
        if mo == 3: add(D(14), "AMAZON PRIME MEMBERSHIP", "PRIME ANNUAL", -139.00, 2)
        for _ in range(int(r.integers(7, 10))):
            add(D(r.integers(1, 29)), pick(["WHOLE FOODS","TRADER JOES","WALMART"]), "GROCERY PURCHASE", -round(r.uniform(30, 140), 2), 2)
        for _ in range(6 + i // 4):
            mer = pick(["CHIPOTLE","BLUE HERON CAFE","QUILL & BEAN ROASTERS"])
            add(D(r.integers(1, 29)), mer, mer + " RESTAURANT CAFE", -round(r.uniform(40, 100) * (1 + .015 * i), 2), 2)
        for _ in range(int(r.integers(6, 10))): add(D(r.integers(1, 29)), "STARBUCKS", "STARBUCKS COFFEE", -round(r.uniform(4, 9), 2), 2)
        for _ in range(int(r.integers(3, 6))): add(D(r.integers(1, 29)), pick(["UBER","LYFT"]), "RIDE", -round(r.uniform(12, 45), 2), 2)
        for _ in range(3): add(D(r.integers(1, 29)), "SHELL", "SHELL FUEL", -round(r.uniform(38, 60), 2), 2)
        for _ in range(int(r.integers(3, 6))): add(D(r.integers(1, 29)), "AMAZON MKTPLACE", "AMAZON ORDER", -round(r.uniform(20, 150), 2), 2)
        for _ in range(int(r.integers(1, 3))): add(D(r.integers(1, 29)), "TARGET", "TARGET STORE", -round(r.uniform(25, 90), 2), 2)
        if r.random() < .6: add(D(r.integers(1, 29)), "CVS PHARMACY", "CVS PHARMACY", -round(r.uniform(15, 60), 2), 2)
        if r.random() < .5: add(D(r.integers(1, 29)), "AMC THEATERS", "MOVIE TICKETS", -round(r.uniform(15, 40), 2), 2)
        if r.random() < .3: add(D(r.integers(1, 29)), "ZORVEX LABS", "ZORVEX LABS ONLINE ORDER STORE", -round(r.uniform(20, 90), 2), 2)
        if mo in (6, 12): add(D(10), "DELTA AIR LINES", "DELTA AIRLINE FLIGHT", -round(r.uniform(380, 420), 2), 2)
    add("2026-10-17", "EMBER & OAK STEAKHOUSE", "EMBER OAK STEAKHOUSE RESTAURANT", -480, 2)   # pattern: unusual
    add("2026-06-09", "AMAZON MKTPLACE", "AMAZON REFUND", 45, 2)                              # edge: refund
    add("2026-08-11", "", "UNKNOWN POS 4412", -23.5, 2)                                       # edge: missing merchant
    rows.append(rows[50])                                                                     # edge: duplicate import
    return rows

def ingest(rows, bid, um=None):
    um = um if um is not None else {x["norm"]: x["category"] for x in Q("select * from user_map")}
    ok = dup = rev = 0; cats = {}; c = con()
    for d, m, desc, cents, acct in rows:
        n = norm(m); h = hashlib.sha1(f"{d}|{m}|{desc}|{cents}|{acct}".encode()).hexdigest()
        cat, conf = categorize(n, f"{desc} {m}", um); rv = int(conf < TH)
        if rv: cat = "Other"
        typ = "income" if cat == "Income" else ("refund" if cents > 0 else "expense")
        try:
            c.execute("insert into transactions(user_id,account_id,merchant,norm,description,amount_cents,transaction_type,transaction_date,"
                      "category,category_confidence,needs_review,import_batch_id,hash) values(1,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (acct, m or "(missing)", n, desc, cents, typ, d, cat, conf, rv, bid, h))
        except sqlite3.IntegrityError: dup += 1; continue
        ok += 1; rev += rv; cats[cat] = cats.get(cat, 0) + 1
    c.commit(); c.close(); return ok, dup, cats, rev

# ---------------- analytics engines ----------------
def df_all():
    d = pd.DataFrame(Q("select * from transactions"))
    d["date"] = pd.to_datetime(d["transaction_date"]); d["cents"] = d["amount_cents"]; return d

def detect(d):
    out = []
    for m, g in d[(d.cents < 0) & (d.category != "Income")].groupby("norm"):
        g = g.sort_values("date").drop_duplicates("date"); dts = g.date.tolist()
        if len(dts) < 2: continue
        iv = np.diff(g.date.values).astype("timedelta64[D]").astype(int); med = float(np.median(iv))
        f = next((x for x in FREQ if x[1] <= med <= x[2]), None)
        if not f or len(dts) < (2 if f[0] == "annual" else 3) or np.max(np.abs(iv - med)) > max(4, .2 * med): continue
        a = (-g.cents.values / 100).round(2); steps = [(str(dts[i].date()), float(a[i-1]), float(a[i])) for i in range(1, len(a))
                                                         if abs(a[i] - a[i-1]) / a[i-1] > .03]
        variable = len(steps) > max(2, len(a) // 4)
        reg = max(0, 1 - float(np.std(iv)) / med); conf = round(min(.99, .5 * reg + .1 * min(len(dts), 5)), 2)
        amt = round(float(np.median(a[-3:])), 2)
        chg = None if (variable or not steps) else {"date": steps[-1][0], "from": steps[-1][1], "to": steps[-1][2],
                                                    "annual_extra": round((steps[-1][2] - steps[-1][1]) * f[3], 2)}
        out.append(dict(merchant=m, kind="bill" if g.category.iloc[0] in ("Housing", "Utilities") else "subscription",
                        billing_frequency=f[0], estimated_amount=amt, monthly_equivalent=round(amt * f[3] / 12, 2),
                        next_expected_payment=str((dts[-1] + f[4]).date()), last_payment=str(dts[-1].date()), confidence=conf,
                        price_change=json.dumps(chg), n=len(dts), history=json.dumps([[str(x.date()), float(y)] for x, y in zip(dts, a)][-24:])))
    return out

def find_anomalies(d):
    out = []; e = d[(d.cents < 0) & (d.recurring == 0) & (~d.category.isin(["Housing", "Income"]))]
    for cat, g in e.groupby("category"):
        if len(g) < 10: continue
        a = -g.cents / 100; med = a.median(); mad = (a - med).abs().median() or 1e-9; z = .6745 * (a - med) / mad
        for i in g.index[(a >= 3 * med) & (z > 3.5)]:
            out.append(dict(transaction_id=int(g.id[i]), anomaly_score=round(min(1, float(z[i]) / 10), 2), detection_method="robust z-score (median/MAD) vs category history",
                            user_feedback="Needs Review",
                            explanation=f"This transaction is {a[i]/med:.1f}x your typical {cat.lower()} purchase (median ${med:.2f} across {len(g)} comparable transactions). Unusual is not the same as fraud."))
    return out

def wavg(s): return float(np.dot(s.iloc[-3:].values, [.2, .3, .5][-len(s.iloc[-3:]):]))
def forecast(d, subs):
    e = d[d.category != "Income"]; per = e.date.dt.to_period("M")
    m = -e.groupby(per).cents.sum() / 100; rc = (-e[e.recurring == 1].groupby(per).cents.sum() / 100).reindex(m.index, fill_value=0)
    v = m - rc; nxt = m.index[-1] + 1
    if len(m) < 10: return None
    errs = [abs(rc.iloc[i] + wavg(v.iloc[:i]) - m.iloc[i]) for i in range(len(m) - 6, len(m))]
    naive = [abs(m.iloc[i-1] - m.iloc[i]) for i in range(len(m) - 6, len(m))]
    up = []
    for s in subs:
        if s["status"] == "Cancelled": continue
        off = next(x[4] for x in FREQ if x[0] == s["billing_frequency"]); t = pd.Timestamp(s["last_payment"]) + off
        while t <= nxt.end_time:
            if t >= nxt.start_time: up.append(dict(date=str(t.date()), merchant=s["merchant"], amount=s["estimated_amount"], kind=s["kind"]))
            t += off
    rec = sum(x["amount"] for x in up); disc = wavg(v); pred = rec + disc; sd = float(np.sqrt(np.mean(np.square(errs))))
    inc = float(d[d.category == "Income"].groupby(d.date.dt.to_period("M")).cents.sum().iloc[-6:].mean() / 100)
    METRICS["forecast_mae_6_holdout_months"] = round(float(np.mean(errs)), 2); METRICS["naive_last_month_mae"] = round(float(np.mean(naive)), 2)
    return dict(period=str(nxt), predicted=round(pred, 2), lower=round(max(0, pred - 1.28 * sd), 2), upper=round(pred + 1.28 * sd, 2),
                recurring=round(rec, 2), discretionary=round(disc, 2), income=round(inc, 2), net=round(inc - pred, 2),
                mae=METRICS["forecast_mae_6_holdout_months"], naive_mae=METRICS["naive_last_month_mae"], upcoming=sorted(up, key=lambda x: x["date"]))

def recs(d, subs, fc):
    out = []; act = [s for s in subs if s["status"] != "Cancelled" and s["kind"] == "subscription"]
    ov = set()
    for g, mem in OVERLAP.items():
        hit = sorted([s for s in act if s["merchant"] in mem], key=lambda s: s["monthly_equivalent"])
        if len(hit) >= 2:
            lo = hit[0]; ov.add(lo["merchant"]); mo = lo["monthly_equivalent"]
            out.append(dict(key=f"overlap:{g}", recommendation_type="subscription_review", title=f"Review overlapping {g} services",
                description=f"{', '.join(h['merchant'].title() for h in hit)} may serve a similar purpose (not duplicate charges). Savings assume cancelling only the cheaper one.",
                monthly_savings_estimate=mo, annual_savings_estimate=round(mo * 12, 2), confidence=.6, action=f"Decide whether you need both; consider cancelling {lo['merchant'].title()}.",
                evidence=[h["merchant"] for h in hit], counted=1))
    for s in act:
        c = json.loads(s["price_change"]) if s["price_change"] else None
        if c and s["merchant"] not in ov and c["annual_extra"] > 0:
            out.append(dict(key=f"price:{s['merchant']}", recommendation_type="price_increase", title=f"{s['merchant'].title()} price increased",
                description=f"Price rose from ${c['from']:.2f} to ${c['to']:.2f} on {c['date']}; extra annualized cost shown.", monthly_savings_estimate=round(c["annual_extra"] / 12, 2),
                annual_savings_estimate=c["annual_extra"], confidence=.9, action="Check for a cheaper plan or negotiate; or cancel if unused.", evidence=c, counted=1))
    rs = d[(d.category == "Restaurants") & (d.cents < 0)]; ms = (-rs.groupby(rs.date.dt.to_period("M")).cents.sum() / 100).iloc[-6:]
    if len(ms) == 6 and ms.iloc[3:].mean() > 1.1 * ms.iloc[:3].mean():
        l3 = float(ms.iloc[3:].mean()); pct = (l3 / float(ms.iloc[:3].mean()) - 1) * 100
        out.append(dict(key="trend:restaurants", recommendation_type="spending_reduction", title="Restaurant spending is trending up",
            description=f"Last-3-month average ${l3:.2f}/mo is {pct:.0f}% above the prior 3 months. Scenario: reduce by 15%.", monthly_savings_estimate=round(.15 * l3, 2),
            annual_savings_estimate=round(.15 * l3 * 12, 2), confidence=.55, action="Set a monthly dining budget.", evidence=[round(float(x), 2) for x in ms], counted=1))
    if fc and fc["upcoming"]:
        early = sum(u["amount"] for u in fc["upcoming"] if int(u["date"][-2:]) <= 7)
        if early / max(1, fc["recurring"]) >= .4:
            out.append(dict(key="cashflow:early", recommendation_type="cash_flow", title=f"{early/fc['recurring']*100:.0f}% of next month's bills land in days 1-7",
                description=f"${early:.2f} of ${fc['recurring']:.2f} scheduled bills are predicted in the first week of {fc['period']}.", monthly_savings_estimate=0, annual_savings_estimate=0,
                confidence=.8, action="Keep a buffer before the 1st, or move due dates if your providers allow.", evidence=fc["upcoming"][:8], counted=0))
    return out

def rebuild():
    ps = {r["merchant"]: r["status"] for r in Q("select merchant,status from subscriptions")}
    pa = {r["transaction_id"]: r["user_feedback"] for r in Q("select transaction_id,user_feedback from anomalies")}
    pr = {r["key"]: r["status"] for r in Q("select key,status from recommendations")}
    for t in ("subscriptions", "anomalies", "recommendations", "forecasts"): X(f"delete from {t}")
    d = df_all(); subs = detect(d)
    for s in subs: s["status"] = ps.get(s["merchant"], "Keep"); INS("subscriptions", dict(user_id=1, **s))
    X("update transactions set recurring=0")
    for s in subs: X("update transactions set recurring=1 where norm=? and amount_cents<0", (s["merchant"],))
    d = df_all()
    for a in find_anomalies(d): a["user_feedback"] = pa.get(a["transaction_id"], a["user_feedback"]); INS("anomalies", a)
    fc = forecast(d, subs)
    if fc: INS("forecasts", dict(user_id=1, forecast_period=fc["period"], predicted_amount=fc["predicted"], lower_bound=fc["lower"], upper_bound=fc["upper"],
                                  model_version="recurring+wavg-v1", detail=json.dumps(fc)))
    for r in recs(d, subs, fc):
        r["status"] = pr.get(r["key"], "New"); r["evidence"] = json.dumps(r["evidence"]); INS("recommendations", dict(user_id=1, **r))

def seed():
    t = time.time()
    c = con(); c.close()
    for t_ in ("transactions", "subscriptions", "anomalies", "forecasts", "recommendations", "import_batches", "user_map", "users", "accounts"):
        X(f"drop table if exists {t_}")
    c = con(); c.executescript(SCHEMA)
    c.execute("insert into users values(1,'Avni Kotharkar','avni@demo.spendsense.app',datetime('now'))")
    c.execute("insert into accounts values(1,1,'Everyday Checking','checking','USD'),(2,1,'Rewards Card','credit_card','USD')"); c.commit(); c.close()
    rows = gen(); X("insert into import_batches(user_id) values(1)")
    ok, dup, cats, rev = ingest(rows, 1, {}); rebuild()
    X("update import_batches set records_received=?,records_imported=?,duplicates_skipped=?,records_rejected=0 where id=1", (len(rows), ok, dup))
    METRICS.update(seed_seconds=round(time.time() - t, 2), seed_records=ok, seed_duplicates_skipped=dup, seed_needs_review=rev)

# ---------------- API ----------------
app = FastAPI(title="SpendSense")
if not os.path.exists(DB) or not Q("select name from sqlite_master where name='transactions'"): seed()
elif not METRICS.get("seed_seconds"): rebuild()

@app.get("/", response_class=HTMLResponse)
def index(): return open(os.path.join(HERE, "index.html"), encoding="utf-8").read()

def pct(a, b): return None if not b else round((a - b) / abs(b) * 100, 1)
@app.get("/api/analytics/overview")
def overview(months: int = 6):
    months = max(1, min(months, 22)); d = df_all(); end = d.date.max(); start = (end.to_period("M") - (months - 1)).start_time
    ps = (start.to_period("M") - months).start_time; cur = d[d.date >= start]; prev = d[(d.date >= ps) & (d.date < start)]
    t = lambda x: dict(spend=round(float(-x[x.category != "Income"].cents.sum() / 100), 2), income=round(float(x[x.category == "Income"].cents.sum() / 100), 2))
    c, p = t(cur), t(prev); sv = Q("select sum(annual_savings_estimate) s from recommendations where counted=1 and status!='Dismissed'", one=True)["s"] or 0
    cb = -cur[cur.category != "Income"].groupby("category").cents.sum() / 100; cb = cb[cb > 0].sort_values(ascending=False)
    return dict(as_of=str(end.date()), period=dict(start=str(start.date()), end=str(end.date())), spending=c["spend"], spending_change_pct=pct(c["spend"], p["spend"]),
                income=c["income"], income_change_pct=pct(c["income"], p["income"]), net=round(c["income"] - c["spend"], 2), potential_annual_savings=round(sv, 2),
                category_breakdown=[dict(category=k, amount=round(float(v), 2), pct=round(float(v / cb.sum() * 100), 1)) for k, v in cb.items()],
                trends=spending_trends(months), insights=insights(d, end))

@app.get("/api/analytics/spending-trends")
def spending_trends(months: int = 22):
    d = df_all(); per = d.date.dt.to_period("M"); inc = d[d.category == "Income"].groupby(per).cents.sum() / 100
    exp = -d[d.category != "Income"].groupby(per).cents.sum() / 100; ix = exp.index[-months:]
    return [dict(month=str(i), income=round(float(inc.get(i, 0)), 2), expenses=round(float(exp[i]), 2), net=round(float(inc.get(i, 0) - exp[i]), 2)) for i in ix]

@app.get("/api/analytics/category-breakdown")
def category_breakdown(months: int = 6): return overview(months)["category_breakdown"]

# chart payloads: {type: bar|hbar|line, labels, values, highlight: index of the emphasized point, caption}
def chart(type, labels, values, highlight=None, caption="", stepped=False):
    return dict(type=type, labels=[str(x) for x in labels], values=[round(float(v), 2) for v in values], highlight=highlight, caption=caption, stepped=stepped)

def anomaly_chart(d, cat, amount):
    e = -d[(d.category == cat) & (d.cents < 0) & (d.recurring == 0)].cents / 100
    return chart("bar", ["Typical (median)", "Upper normal (90th pct)", "This charge"], [e.median(), e.quantile(.9), amount], 2,
                 f"Your {cat.lower()} purchases vs this charge ({len(e)} transactions)")

@app.get("/api/analytics/insights")
def insights_api(): d = df_all(); return insights(d, d.date.max())

def insights(d, end):
    out = []; r = d[(d.category == "Restaurants") & (d.cents < 0)]; ms = -r.groupby(r.date.dt.to_period("M")).cents.sum() / 100
    if len(ms) > 1 and ms.iloc[-2] > 0:
        ch = pct(ms.iloc[-1], ms.iloc[-2]); m6 = ms.iloc[-6:]
        out.append(dict(title=f"Dining spending {'rose' if ch > 0 else 'fell'} {abs(ch)}% vs last month", body=f"${ms.iloc[-1]:.2f} in {ms.index[-1]} vs ${ms.iloc[-2]:.2f} in {ms.index[-2]}.", confidence="calculated", action="Review restaurant transactions.", link="transactions:Restaurants",
                        chart=chart("bar", m6.index, m6.values, len(m6) - 1, "Restaurant spending per month")))
    for s in Q("select * from subscriptions where price_change!='null' and kind='subscription'"):
        c = json.loads(s["price_change"]); h = json.loads(s["history"])[-12:]
        hi = next((i for i, x in enumerate(h) if x[0] >= c["date"]), None)
        out.append(dict(title=f"{s['merchant'].title()} increased in price", body=f"${c['from']:.2f} -> ${c['to']:.2f} on {c['date']}; about ${c['annual_extra']:.2f} more per year.", confidence=f"{s['confidence']:.0%} detection", action="Review the plan.", link="subscriptions",
                        chart=chart("line", [x[0][:7] for x in h], [x[1] for x in h], hi, f"{s['merchant'].title()} charge per billing cycle", stepped=True)))
    ren = [s for s in Q("select * from subscriptions where status!='Cancelled'") if 0 <= (pd.Timestamp(s["next_expected_payment"]) - end).days <= 7]
    if ren: out.append(dict(title=f"{len(ren)} recurring charges expected within 7 days", body=", ".join(f"{s['merchant'].title()} ${s['estimated_amount']:.2f} ({s['next_expected_payment']})" for s in ren), confidence="pattern-based", action="Confirm you want these to renew.", link="subscriptions",
                            chart=chart("hbar", [f"{s['merchant'].title()} · {s['next_expected_payment'][5:]}" for s in ren], [s["estimated_amount"] for s in ren], None, "Expected charge amount")))
    a = Q("select a.*,t.merchant,t.category,t.amount_cents,t.transaction_date from anomalies a join transactions t on t.id=a.transaction_id order by anomaly_score desc limit 1", one=True)
    if a: out.append(dict(title=f"Unusual charge at {a['merchant'].title()}", body=a["explanation"], confidence=f"score {a['anomaly_score']}", action="Mark expected or unexpected.", link="anomalies",
                          chart=anomaly_chart(d, a["category"], -a["amount_cents"] / 100)))
    return out[:5]

@app.get("/api/transactions")
def transactions(q: str = "", category: str = "", sort: str = "transaction_date", desc: bool = True, page: int = 1, size: int = 25, review: bool = False):
    sort = sort if sort in ("transaction_date", "amount_cents", "merchant", "category") else "transaction_date"; w, a = ["1=1"], []
    if q: w.append("(merchant like ? or description like ?)"); a += [f"%{q}%"] * 2
    if category: w.append("category=?"); a.append(category)
    if review: w.append("needs_review=1")
    W = " and ".join(w); size = max(1, min(size, 200))
    n = Q(f"select count(*) n from transactions where {W}", a, True)["n"]
    rows = Q(f"select id,merchant,description,amount_cents,transaction_date,category,category_confidence,needs_review,recurring,transaction_type,account_id from transactions where {W} order by {sort} {'desc' if desc else 'asc'},id desc limit ? offset ?", a + [size, (max(page, 1) - 1) * size])
    for r in rows: r["account"] = ACCT[r["account_id"]]; r["amount"] = r["amount_cents"] / 100; r["status"] = "Review" if r["needs_review"] else "Posted"
    return dict(total=n, page=page, items=rows)

@app.get("/api/transactions/categories")
def categories(): return CATS
class CatIn(BaseModel): category: str
@app.patch("/api/transactions/{tid}")
def patch_tx(tid: int, b: CatIn):
    if b.category not in CATS: raise HTTPException(422, "Unknown category")
    t = Q("select norm from transactions where id=?", (tid,), True)
    if not t: raise HTTPException(404, "Transaction not found")
    X("update transactions set category=?,category_confidence=1,needs_review=0 where id=?", (b.category, tid))
    X("insert into user_map values(?,?) on conflict(norm) do update set category=excluded.category", (t["norm"], b.category)); rebuild()
    return dict(ok=True, learned_mapping=t["norm"])

@app.post("/api/transactions/upload")
async def upload(file: UploadFile):
    raw = await file.read()
    if len(raw) > 1_000_000: raise HTTPException(413, "File exceeds 1 MB limit")
    if not (file.filename or "").lower().endswith(".csv"): raise HTTPException(422, "Only .csv files accepted")
    try: rd = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    except UnicodeDecodeError: raise HTTPException(422, "File must be UTF-8")
    if not rd.fieldnames or not {"date", "merchant", "amount"} <= set(rd.fieldnames): raise HTTPException(422, "CSV needs columns: date, merchant, amount (optional: description, account, currency)")
    good, bad = [], []
    for i, row in enumerate(rd, start=2):
        try:
            dtv = pd.to_datetime(row["date"], errors="raise").date().isoformat(); amt = Decimal(str(row["amount"]).replace("$", "").replace(",", "").strip())
            if (row.get("currency") or "USD").upper() != "USD": raise ValueError("only USD supported")
            if abs(amt) > Decimal("1000000"): raise ValueError("amount out of range")
            good.append((dtv, (row["merchant"] or "").strip(), (row.get("description") or row["merchant"] or "").strip(), int((amt * 100).to_integral_value()), 1 if "check" in (row.get("account") or "").lower() else 2))
        except (ValueError, InvalidOperation, TypeError) as e: bad.append(dict(row=i, reason=str(e)[:80]))
    X("insert into import_batches(user_id,records_received) values(1,?)", (len(good) + len(bad),)); bid = Q("select max(id) m from import_batches", one=True)["m"]
    t = time.time(); ok, dup, cats, rev = ingest(good, bid); 
    if ok: rebuild()
    X("update import_batches set records_imported=?,duplicates_skipped=?,records_rejected=? where id=?", (ok, dup, len(bad), bid))
    METRICS["last_upload_seconds"] = round(time.time() - t, 3)
    return dict(imported=ok, duplicates_skipped=dup, invalid=len(bad), invalid_rows=bad[:20], categories=cats, needs_review=rev)

class StatusIn(BaseModel): status: str
@app.get("/api/subscriptions")
def subs():
    s = Q("select * from subscriptions order by monthly_equivalent desc"); d = df_all(); end = d.date.max()
    for x in s: x["price_change"] = json.loads(x["price_change"]); x["history"] = json.loads(x["history"])
    sub = [x for x in s if x["kind"] == "subscription" and x["status"] != "Cancelled"]; m = round(sum(x["monthly_equivalent"] for x in sub), 2)
    over = [[x["merchant"] for x in sub if x["merchant"] in mem] for mem in OVERLAP.values()]
    return dict(monthly_cost=m, annual_cost=round(m * 12, 2), active=len(sub), upcoming_7d=[x["merchant"] for x in sub if 0 <= (pd.Timestamp(x["next_expected_payment"]) - end).days <= 7],
                overlaps=[o for o in over if len(o) > 1], items=s)
@app.get("/api/subscriptions/{sid}")
def sub(sid: int):
    x = Q("select * from subscriptions where id=?", (sid,), True)
    if not x: raise HTTPException(404, "Not found")
    x["price_change"] = json.loads(x["price_change"]); x["history"] = json.loads(x["history"]); x["annualized_cost"] = round(x["monthly_equivalent"] * 12, 2); return x
@app.patch("/api/subscriptions/{sid}")
def patch_sub(sid: int, b: StatusIn):
    if b.status not in ("Keep", "Review", "Cancelled"): raise HTTPException(422, "status must be Keep, Review or Cancelled")
    if not Q("select 1 from subscriptions where id=?", (sid,)): raise HTTPException(404, "Not found")
    X("update subscriptions set status=? where id=?", (b.status, sid)); rebuild(); return dict(ok=True, note="Tracks your decision only; nothing is cancelled at the merchant.")

@app.get("/api/anomalies")
def anoms():
    r = Q("select a.*,t.merchant,t.category,t.transaction_date,-t.amount_cents/100.0 amount from anomalies a join transactions t on t.id=a.transaction_id order by anomaly_score desc")
    d = df_all() if r else None
    for x in r: x["chart"] = anomaly_chart(d, x["category"], x["amount"])
    return r
@app.patch("/api/anomalies/{aid}")
def patch_an(aid: int, b: StatusIn):
    if b.status not in ("Expected", "Unexpected", "Needs Review"): raise HTTPException(422, "invalid feedback")
    if not Q("select 1 from anomalies where id=?", (aid,)): raise HTTPException(404, "Not found")
    X("update anomalies set user_feedback=? where id=?", (b.status, aid)); return dict(ok=True)

@app.get("/api/forecasts")
def forecasts():
    f = Q("select detail from forecasts", one=True)
    if not f: raise HTTPException(409, "Insufficient history (need >= 10 months)")
    return dict(forecast=json.loads(f["detail"]), history=spending_trends(22))
@app.get("/api/forecasts/upcoming-expenses")
def upcoming(): return forecasts()["forecast"]["upcoming"]

@app.get("/api/recommendations")
def get_recs():
    r = Q("select * from recommendations order by counted desc, annual_savings_estimate desc")
    for x in r: x["evidence"] = json.loads(x["evidence"])
    c = [x for x in r if x["counted"]]
    return dict(potential_annual=round(sum(x["annual_savings_estimate"] for x in c if x["status"] != "Dismissed"), 2), opportunities=len(c), reviewed=len([x for x in c if x["status"] != "New"]),
                accepted_annual=round(sum(x["annual_savings_estimate"] for x in c if x["status"] == "Accepted"), 2),
                note="Estimates only. Accepted = tracked intent, not realized savings.", items=r)
@app.patch("/api/recommendations/{rid}")
def patch_rec(rid: int, b: StatusIn):
    if b.status not in ("New", "Reviewing", "Accepted", "Dismissed"): raise HTTPException(422, "invalid status")
    if not Q("select 1 from recommendations where id=?", (rid,)): raise HTTPException(404, "Not found")
    X("update recommendations set status=? where id=?", (b.status, rid)); return dict(ok=True)

# ---- assistant: allowlisted tools only; LLM (if configured) only phrases verified results ----
def t_cat(word=""):
    d = df_all(); e = d[(d.cents < 0) & (d.category != "Income")]; e = e[e.date >= e.date.max() - pd.DateOffset(months=6)]
    g = (-e.groupby("category").cents.sum() / 100).round(2).sort_values(ascending=False); return dict(window="last 6 months", by_category=g.to_dict(), focus=word)
def t_month():
    t = spending_trends(3); return dict(months=t, change_pct=pct(t[-1]["expenses"], t[-2]["expenses"]))
def t_subs(): s = subs(); return dict(monthly_cost=s["monthly_cost"], top=[(x["merchant"], x["monthly_equivalent"]) for x in s["items"] if x["kind"] == "subscription"][:5])
def t_anom(): return [dict(m=a["merchant"], date=a["transaction_date"], amount=a["amount"], why=a["explanation"]) for a in anoms()[:5]]
def t_fc(): f = forecasts()["forecast"]; return {k: f[k] for k in ("period", "predicted", "lower", "upper", "recurring", "discretionary")}
def t_sav(): r = get_recs(); return dict(potential_annual=r["potential_annual"], items=[(x["title"], x["annual_savings_estimate"]) for x in r["items"] if x["counted"]])
class Chat(BaseModel): question: str
@app.post("/api/assistant/chat")
def chat(b: Chat):
    ql = b.question.lower(); w = next((c for c in CATS if c.lower().rstrip("s") in ql), "")
    tool, fn = (("get_savings_opportunities", t_sav) if re.search(r"sav|cut|reduce", ql) else ("get_unusual_transactions", t_anom) if re.search(r"unusual|anomal|strange", ql)
        else ("get_expense_forecast", t_fc) if re.search(r"next month|expect|forecast|predict", ql) else ("get_active_subscriptions", t_subs) if "subscri" in ql
        else ("get_monthly_spending", t_month) if re.search(r"more|why|month", ql) else ("get_spending_by_category", lambda: t_cat(w)))
    ev = fn(); key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            import httpx
            r = httpx.post("https://api.anthropic.com/v1/messages", timeout=30, headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json=dict(model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"), max_tokens=500, system="Explain using ONLY the JSON evidence. Never invent numbers. You have no bank access. Data is synthetic.",
                          messages=[dict(role="user", content=f"Question: {b.question}\nEvidence: {json.dumps(ev, default=str)}")]))
            return dict(mode="llm", tool=tool, answer=r.json()["content"][0]["text"], evidence=ev)
        except Exception: pass
    return dict(mode="demo (no LLM configured: templated answer from real analytics)", tool=tool, answer=f"Result of {tool}: " + json.dumps(ev, default=str)[:700], evidence=ev)

@app.post("/api/demo/reset")
def reset(): seed(); return dict(ok=True, **METRICS)
@app.get("/api/demo/status")
def status(): return dict(synthetic=True, user=Q("select name from users where id=1", one=True)["name"], transactions=Q("select count(*) n from transactions", one=True)["n"], **{k: v for k, v in METRICS.items()})
@app.get("/api/metrics")
def metrics(): return METRICS
