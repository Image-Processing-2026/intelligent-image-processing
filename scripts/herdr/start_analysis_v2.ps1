<#
.SYNOPSIS
  Khởi các lane Analysis v2: mỗi lane một git worktree + một workspace Herdr + một agent Claude Code.

.DESCRIPTION
  Chạy trong một pane Herdr (HERDR_ENV=1), từ repo chính hoặc worktree tích hợp.
  Xem docs/analysis_v2/orchestration.md.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\herdr\start_analysis_v2.ps1            # wave 1 + lead
  powershell -ExecutionPolicy Bypass -File scripts\herdr\start_analysis_v2.ps1 -Lanes calib
  powershell -ExecutionPolicy Bypass -File scripts\herdr\start_analysis_v2.ps1 -Lanes bench -PromptOnly
#>
[CmdletBinding()]
param(
    # Mặc định = wave 1. Wave 2: calib, agent. Wave 3: fusion. Wave 4: color.
    [string[]]$Lanes = @('bench', 'measure', 'region', 'meta'),
    [switch]$NoLead,
    # Chỉ gửi lại prompt khởi động (khi agent kẹt ở hộp thoại trust lúc khởi động)
    [switch]$PromptOnly,
    # Rỗng = dùng $LaneModels
    [string]$Model = '',
    [string]$PermissionMode = 'auto',
    [string]$BaseBranch = 'feat/analysis-v2',
    [string]$WorktreeRoot = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$KnownLanes = @('bench', 'measure', 'region', 'meta', 'calib', 'agent', 'fusion', 'color')
# Model theo lane: ~/.claude/settings.json có thể đặt haiku làm mặc định. -Model ghi đè cho mọi lane.
$LaneModels = @{
    lead = 'opus'; measure = 'opus'; region = 'opus'; calib = 'opus'; agent = 'opus'; fusion = 'opus'; color = 'opus'
    bench = 'sonnet'; meta = 'sonnet'
}

function Invoke-Step([string]$Description, [scriptblock]$Action) {
    Write-Host "-> $Description"
    if (-not $DryRun) { & $Action }
}

function Invoke-HerdrJson([string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    $raw = & herdr @Arguments | Out-String
    if ($LASTEXITCODE -ne 0) { throw "herdr $($Arguments -join ' ') failed (exit $LASTEXITCODE)" }
    return $raw | ConvertFrom-Json
}

function Test-AgentExists([string]$Name) {
    # stderr của lệnh native thành lỗi dừng khi ErrorActionPreference=Stop (PowerShell 5.1)
    $ErrorActionPreference = 'Continue'
    & herdr agent get $Name *> $null
    return ($LASTEXITCODE -eq 0)
}

# powershell -File truyền 'calib,agent' thành một chuỗi: tách theo dấu phẩy
$Lanes = @($Lanes | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

if ($env:HERDR_ENV -ne '1') { throw 'Run this script inside a Herdr pane (HERDR_ENV=1).' }
foreach ($lane in $Lanes) {
    if ($KnownLanes -notcontains $lane) { throw "Unknown lane '$lane'. Known: $($KnownLanes -join ', ')" }
}

# Repo chính = thư mục cha của git common dir (đúng cả khi chạy từ worktree)
$commonDir = (& git rev-parse --path-format=absolute --git-common-dir).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Not inside a git repository.' }
$MainRepo = Split-Path -Parent $commonDir
if (-not $WorktreeRoot) { $WorktreeRoot = Join-Path (Split-Path -Parent $MainRepo) 'iip-wt' }

& git -C $MainRepo rev-parse --verify --quiet $BaseBranch *> $null
if ($LASTEXITCODE -ne 0) { throw "Base branch '$BaseBranch' does not exist. Create it first (see orchestration.md)." }

$DinoPath = Join-Path $MainRepo 'models\segmentation\grounding-dino-tiny'
$SamPath = Join-Path $MainRepo 'models\segmentation\mobile_sam.pt'
if (-not (Test-Path $DinoPath) -or -not (Test-Path $SamPath)) {
    Write-Warning 'Segmentation models missing in the main repo; the region lane cannot run real_model tests.'
}

Write-Host "Main repo:     $MainRepo"
Write-Host "Worktree root: $WorktreeRoot"
Write-Host "Base branch:   $BaseBranch"
if (-not (Test-Path $WorktreeRoot)) { Invoke-Step "create $WorktreeRoot" { New-Item -ItemType Directory -Path $WorktreeRoot | Out-Null } }

function Copy-LocalAssets([string]$Path) {
    # Tệp không nằm trong git mà lane cần: key Gemini, ảnh benchmark, fixture Penn-Fudan
    Invoke-Step "copy .env, data/real, penn_fudan into $Path" {
        $envFile = Join-Path $MainRepo '.env'
        $envTarget = Join-Path $Path '.env'
        if ((Test-Path $envFile) -and -not (Test-Path $envTarget)) { Copy-Item $envFile $envTarget }
        $relPaths = @('data\real', 'tests\fixtures\prompt_segmentation\penn_fudan')
        foreach ($rel in $relPaths) {
            $src = Join-Path $MainRepo $rel
            if (Test-Path $src) {
                # Bộ hold-out không bao giờ được chép sang lane (chỉ lead chấm trên đó)
                & robocopy $src (Join-Path $Path $rel) /E /XC /XN /XO /XD holdout /NFL /NDL /NJH /NJS /NP | Out-Null
                if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $rel" }
            }
        }
    }
}

function Initialize-Worktree([string]$Name, [string]$Branch, [bool]$NewBranch) {
    $path = Join-Path $WorktreeRoot $Name
    if (Test-Path $path) {
        Write-Host "   worktree $path exists, reusing"
    } else {
        & git -C $MainRepo rev-parse --verify --quiet $Branch *> $null
        $branchExists = ($LASTEXITCODE -eq 0)
        if ($NewBranch -and -not $branchExists) {
            Invoke-Step "git worktree add -b $Branch $path $BaseBranch" {
                & git -C $MainRepo worktree add -b $Branch $path $BaseBranch | Out-Host
                if ($LASTEXITCODE -ne 0) { throw "git worktree add failed for $Name" }
            }
        } else {
            Invoke-Step "git worktree add $path $Branch" {
                & git -C $MainRepo worktree add $path $Branch | Out-Host
                if ($LASTEXITCODE -ne 0) { throw "git worktree add failed for $Name" }
            }
        }
    }
    Copy-LocalAssets $path
    return $path
}

function Start-LaneAgent([string]$Name, [string]$Path, [string]$Kickoff) {
    if ($PromptOnly) {
        Invoke-Step "prompt $Name" { & herdr agent prompt $Name $Kickoff | Out-Null }
        return
    }
    if (Test-AgentExists $Name) {
        Write-Host "   agent '$Name' already running, skipping"
        return
    }
    $script:workspace = $null
    Invoke-Step "herdr workspace av2-$Name" {
        $script:workspace = Invoke-HerdrJson @(
            'workspace', 'create', '--cwd', $Path, '--label', "av2-$Name", '--no-focus',
            '--env', "REGION_DINO_MODEL_PATH=$DinoPath",
            '--env', "REGION_MOBILE_SAM_CHECKPOINT=$SamPath",
            '--env', 'PYTHONUTF8=1'
        )
    }
    if ($DryRun) { return }
    $paneId = $script:workspace.result.root_pane.pane_id
    if (-not $paneId) { throw "Could not read root pane id for $Name" }

    # Ghép '--' trong mảng: Windows PowerShell có thể nuốt token '--' viết trực tiếp
    $startArgs = @('agent', 'start', $Name, '--kind', 'claude', '--pane', $paneId, '--timeout', '120000',
        '--', '--permission-mode', $PermissionMode)
    $laneModel = $LaneModels[$Name]
    if ($Model) { $laneModel = $Model }
    if ($laneModel) { $startArgs += @('--model', $laneModel) }
    Write-Host "   starting '$Name' with model '$laneModel'"
    & herdr @startArgs | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning ("Agent '$Name' is not ready (often the folder-trust dialog). Answer it in workspace " +
            "av2-$Name, then run: start_analysis_v2.ps1 -Lanes $Name -PromptOnly")
        return
    }
    & herdr agent prompt $Name $Kickoff | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Warning "Prompt to '$Name' failed; rerun with -Lanes $Name -PromptOnly" }
    else { Write-Host "   '$Name' started in pane $paneId" }
}

# Lead: worktree tích hợp trên chính nhánh base, chỉ khởi cùng wave 1
if (-not $NoLead -and -not $PromptOnly -and ($Lanes -contains 'bench')) {
    $leadPath = Initialize-Worktree 'integration' $BaseBranch $false
    $leadKickoff = 'You are the lead for Analysis v2. Read docs/analysis_v2/agents/lead.md and follow it. ' +
        'Wave-1 lanes (bench, measure, region, meta) are starting now; monitor them with herdr and report to the user.'
    Start-LaneAgent 'lead' $leadPath $leadKickoff
}

foreach ($lane in $Lanes) {
    $branch = "feat/analysis-v2-$lane"
    $path = Initialize-Worktree $lane $branch $true
    $kickoff = "You are the '$lane' lane agent for Analysis v2, working in this worktree on branch $branch. " +
        "Read docs/analysis_v2/agents/$lane.md and follow it exactly, starting with the documents it lists. " +
        "Do not push, open PRs or merge. Keep docs/analysis_v2/status/$lane.md updated and end it with LANE DONE when finished."
    Start-LaneAgent $lane $path $kickoff
}

Write-Host ''
Write-Host 'Done. Check with: herdr agent list'
