"""数据库层测试（10 号票六表 + schema_version）。"""

from plat.db.models import AgentDecision, Deployment, EpochMetric, Run, Task, init_db


def test_init_db_creates_six_tables(tmp_path):
    db = tmp_path / "platform.db"
    session_factory = init_db(f"sqlite:///{db}")
    with session_factory() as s:
        t = Task(uid="t1" * 4, name="demo", base_model="yolo11n", train_args={"epochs": 5})
        s.add(t)
        s.flush()
        r = Run(uid="r1" * 4, task_id=t.id, round_no=1)
        s.add(r)
        s.flush()
        s.add(EpochMetric(run_id=r.id, epoch=1, metrics={"map50": 0.3}))
        s.add(AgentDecision(run_id=r.id, round_no=1, response={"action": "tune"}))
        s.add(Deployment(run_id=r.id, model_lib_dir="/models/x", onnx_path="/models/x/model.onnx"))
        s.commit()

        assert s.query(Task).count() == 1
        assert s.query(Run).count() == 1
        assert s.query(EpochMetric).count() == 1
        assert s.query(AgentDecision).count() == 1
        assert s.query(Deployment).count() == 1
        row = s.execute(__import__("sqlalchemy").text("SELECT version FROM schema_version")).first()
        assert row.version == 1


def test_init_db_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path}/p.db"
    init_db(url)
    init_db(url)  # 二次调用不炸、不重复
    session_factory = init_db(url)
    with session_factory() as s:
        assert s.query(Task).count() == 0
