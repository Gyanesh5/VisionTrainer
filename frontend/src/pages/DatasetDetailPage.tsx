import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { datasetsApi, TASK_TYPE_ZH } from "../api/client";

/**
 * 数据集详情路由壳（票 109 批 1 落位，批 2 起填工作区）：
 * 组合视图 meta + 徽章行；状态卡与工作区（类别表/校验/体检/审查/切片）按批次移植。
 */
export function DatasetDetailPage() {
  const { name = "" } = useParams();
  const metaQ = useQuery({
    queryKey: ["dataset", name],
    queryFn: () => datasetsApi.get(name),
    enabled: name !== "",
  });
  const meta = metaQ.data;

  return (
    <div className="page-body">
      <div className="page-head">
        <Link to="/datasets" className="crumb-link">数据集</Link>
        <span aria-hidden="true"> / </span>
        <h1 className="crumb-current">{name}</h1>
      </div>
      {metaQ.isError ? (
        <div className="card">
          <p className="error-note">数据集读取失败：{(metaQ.error as Error).message}</p>
          <button type="button" className="btn" onClick={() => void metaQ.refetch()}>重试</button>
        </div>
      ) : metaQ.isPending ? (
        <p className="placeholder-note">读取数据集详情…</p>
      ) : meta ? (
        <div className="card" data-testid="detail-summary">
          <p className="meta-row">
            <span className="meta-label">任务类型</span>
            <span className="meta-value">
              {TASK_TYPE_ZH[meta.task_type]} · {meta.form === "managed" ? "托管" : "外部"}
            </span>
          </p>
          {meta.form === "managed" ? (
            <>
              <p className="meta-row">
                <span className="meta-label">类别表</span>
                <span className="meta-value tnum">
                  {meta.classes?.length ?? 0} 类（{(meta.classes ?? []).join("、") || "尚未登记"}）
                </span>
              </p>
              <p className="meta-row">
                <span className="meta-label">校验</span>
                <span className="meta-value">
                  {meta.validation_summary?.ok === true
                    ? "通过"
                    : meta.validation_summary?.ok === false
                      ? "有问题"
                      : "未校验"}
                </span>
              </p>
            </>
          ) : (
            <p className="meta-row">
              <span className="meta-label">外部路径</span>
              <span className="meta-value">{meta.path}（平台只读，永不改写）</span>
            </p>
          )}
          <p className="placeholder-note">状态卡与工作区（类别表 / 校验·画像 / 体检 / 审查）按票 109 批 2–5 移植。</p>
        </div>
      ) : null}
    </div>
  );
}
