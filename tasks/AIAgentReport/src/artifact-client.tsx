import { getAccessToken, getHost } from "azure-devops-extension-sdk";
import JSZip from "jszip";

const ARTIFACT_NAME = "ai-agent-eval";
const ARTIFACT_FILE = "ai-agent-eval-summary.md";
const FILE_PATH = `ai-agent-eval/${ARTIFACT_FILE}`;

export const fetchArtifactContent = async (
  buildId: number,
  projectName: string,
): Promise<string> => {
  try {
    const accessToken = await getAccessToken();
    const host = await getHost();

    const apiUrl = `https://dev.azure.com/${host.name}/${projectName}/_apis/build/builds/${buildId}/artifacts?artifactName=${ARTIFACT_NAME}&api-version=7.1&%24format=zip`;
    console.log("Fetching artifact from URL:", apiUrl);

    const response = await fetch(apiUrl, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      redirect: "follow",
    });

    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.statusText}`);
    }

    const blob = await response.blob();
    const zip = await JSZip.loadAsync(blob);
    const file = zip.file(FILE_PATH);

    if (!file) {
      throw new Error(`File ${FILE_PATH} not found in the artifact`);
    }

    return await file.async("string");
  } catch (error) {
    console.error("Error fetching artifact:", error);
    throw error;
  }
};
