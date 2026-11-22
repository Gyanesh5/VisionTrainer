import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router-dom";
import { AppShell } from "./shell/AppShell";
import { OverviewPage } from "./pages/OverviewPage";
import { DatasetsPage } from "./pages/DatasetsPage";
import { DatasetDetailPage } from "./pages/DatasetDetailPage";
import { TrainingPage } from "./pages/TrainingPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { RunAnalysisPage } from "./pages/RunAnalysisPage";
import { ComparePage } from "./pages/ComparePage";
import { ModelsPage } from "./pages/ModelsPage";
import { AgentPage } from "./pages/AgentPage";
import { LogsPage } from "./pages/LogsPage";
import { SystemPage } from "./pages/SystemPage";

/** v2 导航路由（Penpot「VisionTrainer v2」定稿）：/ 回落到总览；对比页留训练域
 * 不升导航（map §5 裁决）；日志退出侧导航但路由保留（入口在系统页）；
 * /deploy 页面为模型页（票 121），路由路径保持不变只换门牌。 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell><Outlet /></AppShell>}>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:name" element={<DatasetDetailPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/training/runs/:uid" element={<RunDetailPage />} />
          <Route path="/training/runs/:uid/analysis" element={<RunAnalysisPage />} />
          <Route path="/training/compare" element={<ComparePage />} />
          <Route path="/deploy" element={<ModelsPage />} />
          <Route path="/agent" element={<AgentPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route path="/system" element={<SystemPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
