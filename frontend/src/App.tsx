import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router-dom";
import { AppShell } from "./shell/AppShell";
import { DatasetsPage } from "./pages/DatasetsPage";
import { DatasetDetailPage } from "./pages/DatasetDetailPage";
import { TrainingPage } from "./pages/TrainingPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { ComparePage } from "./pages/ComparePage";
import { DeployPage } from "./pages/DeployPage";
import { AgentPage } from "./pages/AgentPage";
import { LogsPage } from "./pages/LogsPage";
import { SystemPage } from "./pages/SystemPage";

/** 六导航路由（票 107）：/ 回落到数据集；对比页留训练域不升导航（map §5 裁决）。 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell><Outlet /></AppShell>}>
          <Route path="/" element={<Navigate to="/datasets" replace />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:name" element={<DatasetDetailPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/training/runs/:uid" element={<RunDetailPage />} />
          <Route path="/training/compare" element={<ComparePage />} />
          <Route path="/deploy" element={<DeployPage />} />
          <Route path="/agent" element={<AgentPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route path="/system" element={<SystemPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
