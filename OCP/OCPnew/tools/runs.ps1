param(
    [Parameter(Mandatory = $true)][string]$Path,
    [int]$Threshold = 128,
    [int]$Y0 = 0, [int]$Y1 = -1, [int]$Step = 2,
    [double]$CX = -1, [double]$CY = -1
)

Add-Type -AssemblyName System.Drawing

$bmp = New-Object System.Drawing.Bitmap $Path
$w = $bmp.Width; $h = $bmp.Height
$rect = New-Object System.Drawing.Rectangle 0, 0, $w, $h
$bd = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$stride = $bd.Stride
$bytes = New-Object byte[] ($stride * $h)
[System.Runtime.InteropServices.Marshal]::Copy($bd.Scan0, $bytes, 0, $bytes.Length)
$bmp.UnlockBits($bd)
$bmp.Dispose()

if ($Y1 -lt 0) { $Y1 = $h - 1 }

for ($y = $Y0; $y -le $Y1; $y += $Step) {
    $row = @()
    $inRun = $false; $s = 0
    for ($x = 0; $x -lt $w; $x++) {
        $i = $y * $stride + $x * 3
        $lum = 0.114 * $bytes[$i] + 0.587 * $bytes[$i + 1] + 0.299 * $bytes[$i + 2]
        $ink = ($lum -lt $Threshold)
        if ($ink -and -not $inRun) { $inRun = $true; $s = $x }
        elseif (-not $ink -and $inRun) { $inRun = $false; $row += "$s-$($x-1)" }
    }
    if ($inRun) { $row += "$s-$($w-1)" }
    $extra = ""
    if ($CX -ge 0 -and $CY -ge 0) { $extra = "  dy=$([math]::Round($y - $CY,1))" }
    Write-Output ("y={0,4}{1}  black: {2}" -f $y, $extra, ($row -join "  "))
}
