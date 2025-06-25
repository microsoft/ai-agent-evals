import * as tl from "azure-pipelines-task-lib/task";
import * as path from "path";
import * as fs from "fs";

async function findPythonExecutable(): Promise<string> {
  // Try to find Python on the system
  let pythonPath = tl.which("python", false);
  if (!pythonPath) {
    pythonPath = tl.which("python3", false);
  }

  if (!pythonPath) {
    throw new Error("Python command not found in PATH");
  }

  // Log which Python executable we're using
  console.log(`Using Python executable: ${pythonPath}`);
  return pythonPath;
}

async function checkPython(pythonPath: string): Promise<boolean> {
  const minimumPythonMajor = 3;
  const minimumPythonMinor = 10;

  try {
    console.log("Checking Python installation...");

    // Create a tool runner for Python version check
    const pythonRunner = tl.tool(pythonPath);
    pythonRunner.arg("--version");

    const result = await pythonRunner.exec();

    if (result !== 0) {
      throw new Error("Failed to get Python version");
    }

    // Since --version output goes to stderr, we need to capture it differently
    const pythonVersion = tl.execSync(pythonPath, ["--version"]);
    const versionMatch = pythonVersion.stdout.match(
      /Python (\d+)\.(\d+)\.(\d+)/
    );
    if (!versionMatch) {
      throw new Error("Unable to determine Python version");
    }

    const major = parseInt(versionMatch[1]);
    const minor = parseInt(versionMatch[2]);
    const patch = parseInt(versionMatch[3]);

    if (
      major > minimumPythonMajor ||
      (major === minimumPythonMajor && minor >= minimumPythonMinor)
    ) {
      console.log(
        `Python version ${major}.${minor}.${patch} meets requirements (minimum ${minimumPythonMajor}.${minimumPythonMinor})`
      );
      return true;
    } else {
      console.log(
        `Python version ${major}.${minor}.${patch} found, but version ${minimumPythonMajor}.${minimumPythonMinor} or higher is required`
      );
      throw new Error("Insufficient Python version");
    }
  } catch (error: any) {
    console.log(`Python version check failed: ${error.message}`);
    console.log("");
    console.log("Installation options:");
    console.log(
      `1. Install Python ${minimumPythonMajor}.${minimumPythonMinor}+ from https://www.python.org/downloads/`
    );
    console.log("2. In Azure DevOps pipeline, use:");
    console.log("   ```yaml");
    console.log("   - task: UsePythonVersion@0");
    console.log("     inputs:");
    console.log("       versionSpec: '3.10'");
    console.log("   ```");
    console.log("");
    console.log(
      "After installation, ensure Python is in your PATH and try again."
    );
    return false;
  }
}

async function installPythonDependencies(
  pythonPath: string,
  scriptDir: string
): Promise<void> {
  console.log("Installing Python dependencies...");

  // Upgrade pip
  const pipUpgradeRunner = tl.tool(pythonPath);
  pipUpgradeRunner.arg(["-m", "pip", "install", "--upgrade", "pip"]);

  const pipUpgradeResult = await pipUpgradeRunner.exec();
  if (pipUpgradeResult !== 0) {
    throw new Error("Failed to upgrade pip");
  }

  // Install the package
  const pipInstallRunner = tl.tool(pythonPath);
  pipInstallRunner.arg(["-m", "pip", "install", scriptDir]);

  const pipInstallResult = await pipInstallRunner.exec();
  if (pipInstallResult !== 0) {
    throw new Error("Failed to install Python dependencies");
  }

  console.log("Dependencies installed successfully");
}

async function run() {
  try {
    console.log("Starting AIAgentEvaluation v2 task");
    const scriptDir = __dirname;

    // Find Python executable
    const pythonPath = await findPythonExecutable();

    // Check Python installation
    const pythonCheck = await checkPython(pythonPath);
    if (!pythonCheck) {
      tl.setResult(
        tl.TaskResult.Failed,
        "Python installation check failed. Cannot proceed."
      );
      return;
    }

    // Install Python dependencies
    await installPythonDependencies(pythonPath, scriptDir);

    // Read task inputs
    console.log("Reading task inputs...");

    const inputs = [
      { name: "azure-aiproject-connection-string", required: true },
      { name: "deployment-name", required: true },
      { name: "api-version", required: false },
      { name: "data-path", required: true },
      { name: "agent-ids", required: true },
      { name: "baseline-agent-id", required: false },
      { name: "evaluation-result-view", required: false },
    ];
    inputs.forEach((input) => {
      const value = tl.getInput(input.name, input.required);
      if (value) {
        console.log(`${input.name}: ${value}`);
        process.env[input.name.toUpperCase().replace(/-/g, "_")] = value;
      } else if (input.required) {
        throw new Error(`Required input '${input.name}' is missing`);
      }
    });

    console.log("Executing action.py");
    const artifactFolder = "ai-agent-eval";
    const artifactFile = "ai-agent-eval-summary.md";

    const outputPath = process.env.BUILD_ARTIFACTSTAGINGDIRECTORY || ".";
    const reportPath = path.join(outputPath, artifactFile);

    // Create report file
    fs.writeFileSync(reportPath, "");
    console.log(`Report file created at ${reportPath}`);
    process.env.ADO_STEP_SUMMARY = reportPath; // Execute Python script
    const actionPyPath = path.join(scriptDir, "action.py");
    const pythonRunner = tl.tool(pythonPath);
    pythonRunner.arg(actionPyPath);

    const pythonResult = await pythonRunner.exec();
    if (pythonResult !== 0) {
      throw new Error(`Python script failed with exit code ${pythonResult}`);
    } else {
      console.log("Python script executed successfully");
      console.log(
        `##vso[artifact.upload artifactname=${artifactFolder}]${reportPath}`
      );
    }

    tl.setResult(
      tl.TaskResult.Succeeded,
      `Successfully executed the AIAgentEvaluation task.`
    );
  } catch (err: any) {
    console.log(`An error occurred: ${err.message}`);
    tl.setResult(tl.TaskResult.Failed, err.message);
  }
}

run();
