/** 占位页基线（票 107 骨架期）：页面标题用领域词汇 + 一句状态说明。 */
export function PlaceholderPage({ title, note }: { title: string; note: string }) {
  return (
    <div className="page-body">
      <div className="page-head">
        <h1>{title}</h1>
      </div>
      <p className="placeholder-note">{note}</p>
    </div>
  );
}
