import os, io, tempfile
os.environ["SPENDSENSE_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
from fastapi.testclient import TestClient
import app as A
c = TestClient(A.app)

def test_seed_and_dedupe():
    s = c.get("/api/demo/status").json()
    assert s["transactions"] > 1000 and s["seed_duplicates_skipped"] == 1

def test_subscription_detection_and_price_increase():
    items = {x["merchant"]: x for x in c.get("/api/subscriptions").json()["items"]}
    nf = items["netflix"]["price_change"]
    assert nf["from"] == 15.99 and nf["to"] == 22.99 and nf["annual_extra"] == 84.0
    assert "amazon prime" in items and items["amazon prime"]["billing_frequency"] == "annual"
    assert not {"whole foods", "starbucks", "uber"} & set(items)

def test_overlap_and_anomaly():
    assert any({"netflix", "hulu"} <= set(o) for o in c.get("/api/subscriptions").json()["overlaps"])
    an = c.get("/api/anomalies").json()
    assert any(a["amount"] == 480 and "x your typical restaurants" in a["explanation"] for a in an)

def test_forecast_and_holdout():
    f = c.get("/api/forecasts").json()["forecast"]
    assert f["lower"] <= f["predicted"] <= f["upper"] and f["mae"] > 0 and f["upcoming"]

def test_recs_persist_and_no_double_count():
    r = c.get("/api/recommendations").json(); keys = [x["title"] for x in r["items"]]
    ov = next(x for x in r["items"] if x["key"].startswith("overlap")); price = [x["key"] for x in r["items"] if x["key"].startswith("price:")]
    assert ov["monthly_savings_estimate"] == 17.99 and "price:hulu" not in price and "price:netflix" in price  # savings only from cheaper service
    rid = r["items"][0]["id"]; assert c.patch(f"/api/recommendations/{rid}", json={"status": "Accepted"}).status_code == 200
    assert c.get("/api/recommendations").json()["accepted_annual"] > 0
    assert c.patch(f"/api/recommendations/{rid}", json={"status": "bogus"}).status_code == 422

def test_upload_validation_and_idempotency():
    csv = "date,merchant,amount\n2026-10-30,Trader Joes,-42.10\nnotadate,Foo,-5\n2026-10-30,Bar,abc\n"
    up = lambda: c.post("/api/transactions/upload", files={"file": ("x.csv", io.BytesIO(csv.encode()), "text/csv")}).json()
    a = up(); b = up()
    assert a["imported"] == 1 and a["invalid"] == 2 and b["imported"] == 0 and b["duplicates_skipped"] == 1
    assert c.post("/api/transactions/upload", files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2"), "text/csv")}).status_code == 422

def test_category_correction_persists():
    t = c.get("/api/transactions?q=ZORVEX&size=1").json()["items"][0]
    assert c.patch(f"/api/transactions/{t['id']}", json={"category": "Entertainment"}).status_code == 200
    assert c.get("/api/transactions?q=ZORVEX&size=1").json()["items"][0]["category"] == "Entertainment"

def test_assistant_fallback_labeled():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    r = c.post("/api/assistant/chat", json={"question": "Which subscriptions cost me the most?"}).json()
    assert r["mode"].startswith("demo") and r["tool"] == "get_active_subscriptions"
