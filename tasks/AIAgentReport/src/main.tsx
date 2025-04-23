import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import {
  init,
  ready,
  getConfiguration,
} from "azure-devops-extension-sdk";
import { Build } from "azure-devops-extension-api/Build";
import { fetchArtifactContent } from "./artifact-client";
import { GithubFlavoredMarkdown } from "./github-flavored-markdown";
import { ErrorMessage } from "./error";

const run = async () => {
  await init();
  await ready();

  const config = getConfiguration();
  config.onBuildChanged(async (build: Build) => {
    try {
      if (!build || !build.id) {
        throw new Error("Build information not available");
      }
      console.log("Build ID:", build.id);
      console.log("Project:", build.project.name);
      const markdownContent = await fetchArtifactContent(
        build.id,
        build.project.name
      );

      createRoot(document.getElementById("root")!).render(
        <StrictMode>
          <GithubFlavoredMarkdown markdownContent={markdownContent} />
        </StrictMode>
      );
    } catch (e) {
      if (e instanceof Error) {
        const err = e as Error;
        createRoot(document.getElementById("root")!).render(
          <StrictMode>
            <ErrorMessage message={err.message} />
          </StrictMode>
        );
        return;
      }
    }
  });
};

run();
