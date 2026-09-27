param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Scale = 4,
    [int]$X0 = 0, [int]$Y0 = 0, [int]$W = -1, [int]$H = -1,
    [switch]$Grid
)

Add-Type -AssemblyName System.Drawing

$src = New-Object System.Drawing.Bitmap $Path
if ($W -lt 0) { $W = $src.Width - $X0 }
if ($H -lt 0) { $H = $src.Height - $Y0 }

$dst = New-Object System.Drawing.Bitmap ($W * $Scale), ($H * $Scale)
$g = [System.Drawing.Graphics]::FromImage($dst)
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::NearestNeighbor
$g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::Half
$g.DrawImage($src, (New-Object System.Drawing.Rectangle 0, 0, ($W * $Scale), ($H * $Scale)), (New-Object System.Drawing.Rectangle $X0, $Y0, $W, $H), [System.Drawing.GraphicsUnit]::Pixel)

if ($Grid) {
    $penMinor = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(70, 0, 128, 255)), 1
    $penMajor = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(160, 255, 0, 0)), 1
    $font = New-Object System.Drawing.Font "Consolas", ([float]($Scale * 2.2))
    $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(200, 200, 0, 0))
    for ($x = 0; $x -lt $W; $x++) {
        $px = $x * $Scale
        $abs = $X0 + $x
        if ($abs % 10 -eq 0) {
            $g.DrawLine($penMajor, $px, 0, $px, $dst.Height)
            $g.DrawString("$abs", $font, $brush, $px, 0)
        }
        elseif ($abs % 5 -eq 0) { $g.DrawLine($penMinor, $px, 0, $px, $dst.Height) }
    }
    for ($y = 0; $y -lt $H; $y++) {
        $py = $y * $Scale
        $abs = $Y0 + $y
        if ($abs % 10 -eq 0) {
            $g.DrawLine($penMajor, 0, $py, $dst.Width, $py)
            $g.DrawString("$abs", $font, $brush, 0, $py)
        }
        elseif ($abs % 5 -eq 0) { $g.DrawLine($penMinor, 0, $py, $dst.Width, $py) }
    }
}

$g.Dispose()
$dst.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$dst.Dispose()
$src.Dispose()
Write-Output "wrote $Out"
