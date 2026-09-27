# Pull the logo out of a live screenshot and re-emit it in the reference
# image's frame (195x195, octagon ink box 7.11..187.23) so it can be diffed
# against ocp-2.jpg with verify_logo's numbers.
param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$CropX0 = 0, [int]$CropY0 = 0, [int]$CropX1 = -1, [int]$CropY1 = -1,
    [int]$Threshold = 128,
    [switch]$Profile
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

if ($CropX1 -lt 0) { $CropX1 = $w - 1 }
if ($CropY1 -lt 0) { $CropY1 = $h - 1 }

function IsInk([int]$x, [int]$y) {
    $i = $y * $stride + $x * 3
    $b = $bytes[$i]; $g = $bytes[$i + 1]; $r = $bytes[$i + 2]
    $lum = 0.114 * $b + 0.587 * $g + 0.299 * $r
    if ($lum -ge $Threshold) { return $false }
    # Logo ink is neutral black; reject saturated UI chrome.
    $mx = [math]::Max($r, [math]::Max($g, $b))
    $mn = [math]::Min($r, [math]::Min($g, $b))
    return (($mx - $mn) -lt 40)
}

if ($Profile) {
    for ($y = $CropY0; $y -le $CropY1; $y++) {
        $lo = -1; $hi = -1; $n = 0
        for ($x = $CropX0; $x -le $CropX1; $x++) {
            if (IsInk $x $y) { if ($lo -lt 0) { $lo = $x }; $hi = $x; $n++ }
        }
        Write-Output ("y={0,4}  ink {1,4}  x[{2}..{3}]" -f $y, $n, $lo, $hi)
    }
    return
}

# ink bbox inside the crop
$minX = $CropX1 + 1; $maxX = -1; $minY = $CropY1 + 1; $maxY = -1
for ($y = $CropY0; $y -le $CropY1; $y++) {
    for ($x = $CropX0; $x -le $CropX1; $x++) {
        if (IsInk $x $y) {
            if ($x -lt $minX) { $minX = $x }
            if ($x -gt $maxX) { $maxX = $x }
            if ($y -lt $minY) { $minY = $y }
            if ($y -gt $maxY) { $maxY = $y }
        }
    }
}

$bw = $maxX - $minX + 1
$bh = $maxY - $minY + 1
Write-Output "live ink bbox x[$minX..$maxX] y[$minY..$maxY]  ${bw}x${bh}"

# Reference frame: octagon ink spans 7.11 .. 187.23 (width 180.12) in a 195 box.
$REF = 195
$refLo = 7.11
$refSpan = 180.12
$srcPerRef = [math]::Max($bw, $bh) / $refSpan

# Paper, not ink: reference pixels whose footprint falls outside the crop keep
# this value. Rows past the octagon's box map onto the HUD, so they must not be
# sampled at all.
$grey = New-Object byte[] ($REF * $REF)
for ($i = 0; $i -lt $grey.Length; $i++) { $grey[$i] = 255 }

for ($py = 0; $py -lt $REF; $py++) {
    for ($px = 0; $px -lt $REF; $px++) {
        # reference pixel centre -> source coords
        $u = (($px + 0.5) - $refLo) * $srcPerRef + $minX
        $v = (($py + 0.5) - $refLo) * $srcPerRef + $minY

        # area average over this reference pixel's footprint in the source
        $half = $srcPerRef * 0.5
        $x0 = [math]::Max($CropX0, [int][math]::Floor($u - $half))
        $x1 = [math]::Min($CropX1, [int][math]::Ceiling($u + $half) - 1)
        $y0 = [math]::Max($CropY0, [int][math]::Floor($v - $half))
        $y1 = [math]::Min($CropY1, [int][math]::Ceiling($v + $half) - 1)

        $ink = 0; $tot = 0
        for ($sy = $y0; $sy -le $y1; $sy++) {
            for ($sx = $x0; $sx -le $x1; $sx++) {
                $tot++
                if (IsInk $sx $sy) { $ink++ }
            }
        }

        if ($tot -gt 0 -and ($ink * 2) -ge $tot) { $grey[$py * $REF + $px] = 0 }
    }
}

$fs = [System.IO.File]::Create($Out)
$hdr = [System.Text.Encoding]::ASCII.GetBytes("P5`n$REF $REF`n255`n")
$fs.Write($hdr, 0, $hdr.Length)
$fs.Write($grey, 0, $grey.Length)
$fs.Close()
Write-Output "wrote $Out ($([math]::Round($srcPerRef,4)) source px per reference px)"
