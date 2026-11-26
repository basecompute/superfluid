# Install superfluid on Windows from a GitHub release.
#
#   irm https://basecompute.co/sf/install.ps1 | iex
#
# Environment:
#   SUPERFLUID_VERSION      a release tag (v0.2.0); default: the latest release
#   SUPERFLUID_PREFIX       where the `superfluid` command goes; default:
#                           %LOCALAPPDATA%\Programs\superfluid (its bin\)
#   SUPERFLUID_HOME         where releases are kept, side by side; default:
#                           %USERPROFILE%\.superfluid (versions\)
#   SUPERFLUID_NO_PATH      set to leave your PATH as it is
#   SUPERFLUID_RELEASE_URL  where the release's files are, in place of GitHub: a mirror's URL,
#                           or a folder (file:///C:/...) holding a release built here
#
# Nothing needs an administrator. The release goes under $SUPERFLUID_HOME\versions, and the
# commands in $SUPERFLUID_PREFIX\bin run the one that is current, so `superfluid self update`
# can install the next beside it and `superfluid self use` can go back. A server that is
# running keeps the version it started with. llama.cpp installs itself the first time a model
# needs it.
& {
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Say($m) { Write-Host "superfluid: $m" }
function Fail($m) { throw "superfluid: $m" }
# A folder (file:///...) is copied from: PowerShell 7's web requests take no file URL.
function Fetch($url, $to) {
  if ($url -like 'file:*') { Copy-Item -LiteralPath ([Uri] $url).LocalPath -Destination $to }
  else { Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $to }
}

$repo = 'basecompute/superfluid'
$version = if ($env:SUPERFLUID_VERSION) { $env:SUPERFLUID_VERSION } else { 'latest' }
$prefix = if ($env:SUPERFLUID_PREFIX) { $env:SUPERFLUID_PREFIX } else { Join-Path $env:LOCALAPPDATA 'Programs\superfluid' }

# The machine's architecture, also from a 32-bit PowerShell on a 64-bit Windows.
$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($arch -ne 'AMD64') {
  Fail "no release for Windows on $arch; build from source (see docs/getting_started/installation.md)"
}
$name = 'superfluid-x86_64-pc-windows-msvc'
$base = if ($env:SUPERFLUID_RELEASE_URL) { $env:SUPERFLUID_RELEASE_URL.TrimEnd('/') }
  elseif ($version -eq 'latest') { "https://github.com/$repo/releases/latest/download" }
  else { "https://github.com/$repo/releases/download/$version" }

$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
if (-not (Test-Path $tar)) { Fail 'tar.exe is required (Windows 10 1803 and later have it)' }

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$tmp = Join-Path ([IO.Path]::GetTempPath()) "superfluid-install-$PID"
try {
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  $archive = Join-Path $tmp "$name.tar.gz"
  Say "downloading $name ($version)"
  try { Fetch "$base/$name.tar.gz" $archive }
  catch { Fail "could not download $base/$name.tar.gz ($($_.Exception.Message))" }
  try { Fetch "$base/$name.tar.gz.sha256" "$archive.sha256" }
  catch { Fail "could not download its checksum ($($_.Exception.Message))" }
  $want = ((Get-Content -Raw "$archive.sha256").Trim() -split '\s+')[0].ToLower()
  $have = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLower()
  if ($want -ne $have) { Fail "checksum mismatch: expected $want, got $have" }

  & $tar -xzf $archive -C $tmp
  if ($LASTEXITCODE -ne 0) { Fail "could not unpack $name.tar.gz" }
  $new = Join-Path $tmp $name
  $exe = Join-Path $new 'bin\superfluid.exe'
  # The release must start here before it takes the place of one that does.
  $said = & $exe --version
  if ($LASTEXITCODE -ne 0 -or -not $said) { Fail 'the release does not start on this machine; nothing was changed' }

  # The release places itself: under versions, made current, with the commands in the prefix.
  $bin = Join-Path $prefix 'bin'
  & $exe self install --from $new --prefix $bin
  if ($LASTEXITCODE -ne 0) { Fail 'the release could not be installed; what was here is as it was' }
  $said = & (Join-Path $bin 'superfluid.exe') --version
  if ($LASTEXITCODE -ne 0 -or -not $said) { Fail "the superfluid in $bin does not start" }

  Say "installed $said to $bin"
  if (-not $env:SUPERFLUID_NO_PATH) {
    # Read and written as the registry holds it, so an entry such as %USERPROFILE%\bin stays one.
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
    $mine = $key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    $dirs = @("$mine" -split ';' | Where-Object { $_ })
    if ($dirs -notcontains $bin) {
      $key.SetValue('Path', (($dirs + $bin) -join ';'), [Microsoft.Win32.RegistryValueKind]::ExpandString)
      # Setting any variable this way tells running programs the environment changed.
      [Environment]::SetEnvironmentVariable('SUPERFLUID_PATH_SET', '1', 'User')
      [Environment]::SetEnvironmentVariable('SUPERFLUID_PATH_SET', $null, 'User')
      Say "added $bin to your PATH: a terminal opened from now on finds superfluid"
    }
    $key.Close()
    if (@($env:Path -split ';') -notcontains $bin) { $env:Path = "$env:Path;$bin" }
  }
  Say 'get started: superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M'
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
}
