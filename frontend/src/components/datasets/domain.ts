/**
 * 数据集域共享文案（旧 UI pages.py 单源移植）：校验问题中文名、修复指引、可自动修码。
 * 判据与账目权威都在服务端——这里只承载展示文案。
 */

/** 校验问题码 → 展示名（validate.py PROBLEM_ZH） */
export const PROBLEM_ZH: Record<string, string> = {
  bad_json: "JSON 解析失败",
  unreadable_image: "图片不可读",
  no_matching_shape: "标注形状与任务类型不符",
  unknown_labels: "含未登记类别",
  drop: "标注被转换丢弃",
  out_of_bounds: "坐标越界（超 ±1% 容差）",
  orphan_json: "孤儿标注（无对应图片）",
  polygon_invalid: "多边形无效（自交/退化）",
  obb_invalid: "旋转框无效（退化/非凸）",
  keypoints_invalid: "关键点坐标为负",
  stem_conflict: "同词干多图撞派生名",
  unprojectable: "标注无法投影（缺尺寸/坏坐标）",
};

/** 恒可自动修复的问题码；obb_invalid 须靠 fixable 标记区分乱序（可修）与退化（人工） */
export const AUTO_FIX_CODES = ["out_of_bounds", "polygon_invalid", "keypoints_invalid", "orphan_json"];

/** 校验规则「错误→原因→修法」三行文案（validate.py GUIDANCE 单源） */
export const GUIDANCE: Record<string, { error: string; cause: string; fix: string }> = {
  unreadable_image: {
    error: "图片无法解码（扩展名与文件头不符，或文件为空/损坏）",
    cause: "文件在拷贝或同步时被截断，或扩展名被改过（例如 jpg 实为 png）",
    fix: "在文件管理器里用看图软件确认该图能否打开；打不开就删掉或重新导出该图，再回平台「重新校验」",
  },
  bad_json: {
    error: "标注 JSON 解析失败（不是合法 JSON）",
    cause: "标注文件被手动编辑坏、同步软件截断，或磁盘写入中断",
    fix: "在 X-AnyLabeling 中重新打开该图并重新保存标注（Ctrl+S）；若无法恢复则删除该 JSON 让图片降级为负样本",
  },
  no_matching_shape: {
    error: "标注形状无一能投影到该数据集的任务类型（如检测集里只有关键点/折线）",
    cause: "该图按另一种任务类型标注过（例如先做了姿态又导入为检测集）——矩形/圆/多边形/旋转框在检测集里是合法的（投影为外接框）",
    fix: "在 X-AnyLabeling 中补画本任务类型的形状（检测=矩形），或把该图归到对应任务的数据集",
  },
  unknown_labels: {
    error: "标注里出现类别表未登记的类别",
    cause: "标注时用了新类别，或类别改名后源 JSON 未同步",
    fix: "在 X-AnyLabeling 中把该标注改成已登记类别；确属新类别则先在数据集页改类别表再重新校验",
  },
  missing_json: {
    error: "无标注 JSON",
    cause: "设计内行为：空标注图按负样本处理，只进 train",
    fix: "无需修复——若本意是正样本，在 X-AnyLabeling 中补画标注即可",
  },
  drop: {
    error: "标注未进入训练派生（转换丢弃）",
    cause: "shape 与任务不匹配、类别未登记、几何非法（零面积框/坏圆/点数不足/obb 越界/pose 无框）——明细见每条的 reason",
    fix: "在 X-AnyLabeling 中修正该 shape（改形状/类别/补框）或确认有意丢弃后到数据集页「确认接受丢弃」；segment 里误用折线工具画的封闭形状可「一键修复」闭合成多边形（点不动）",
  },
  out_of_bounds: {
    error: "标注坐标越界超出 ±1% 容差",
    cause: "框画到图像边界外（缩放/旋转标注常见），转换虽按容差裁剪，超容差部分已失真",
    fix: "数据集概览「一键修复」自动钳制回画布并写回标注；要求精确贴合目标本体则回 X-AnyLabeling 重框",
  },
  orphan_json: {
    error: "孤儿标注：JSON 没有对应图片",
    cause: "图片被删除/改名，或 JSON 拷贝时带了多余文件",
    fix: "数据集概览「一键修复」自动移入隔离区（不删除、可找回）；或手动找回同名图片",
  },
  polygon_invalid: {
    error: "分割多边形无效（自交/重复点/退化环）",
    cause: "手绘多边形时线条交叉，或自动标注产物畸形",
    fix: "数据集概览「一键修复」make_valid 自动修复（自交修复后形状可能变形或分裂）；要求贴合原意则回 X-AnyLabeling 重画",
  },
  obb_invalid: {
    error: "旋转框无效（四点退化/非凸）",
    cause: "四角点共线、面积近似零或顶点顺序混乱",
    fix: "顶点乱序可「一键修复」凸包重排；共线/内点退化无法自动修——回 X-AnyLabeling 重画",
  },
  keypoints_invalid: {
    error: "关键点坐标为负（画布外）",
    cause: "关键点拖出画布左/上边界",
    fix: "数据集概览「一键修复」自动钳回画布内（贴边）",
  },
  stem_conflict: {
    error: "同目录同词干的多个图片文件（如 a.jpg 与 a.png 共存）",
    cause: "图片被工具重编码或复制时留了第二种扩展名；派生标签按词干命名，两张图会撞在同一个 txt 上",
    fix: "删掉其中一种扩展名的文件（保留与标注 JSON 配套的那张；在 X-AnyLabeling 里另存常会再生成第二种），再「重新校验」",
  },
  unprojectable: {
    error: "标注无法投影为训练几何（缺画布尺寸或坐标不是数字）",
    cause: "JSON 被手动改坏、导出工具漏写字段，或坐标写成了字符串/缺项",
    fix: "在 X-AnyLabeling 中重新打开该图并另存标注（会补回宽高），无法打开则删除该 JSON 让图片降级为负样本",
  },
};

/** 切分/导出等时间戳显示（DESIGN.md §4：当天 HH:MM:SS、跨天 MM-DD HH:MM:SS） */
export function formatTs(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  const pad = (n: number) => String(n).padStart(2, "0");
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  return sameDay ? hm : `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${hm}`;
}
