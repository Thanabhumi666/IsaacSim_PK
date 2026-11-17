# Draws report figure 2.2 (RL loop) and figure 3.2 (workflow) with System.Drawing.
#   powershell -File draw_diagrams.ps1 [-outDir ...]
# Must be saved as UTF-8 with BOM (Thai text).
param([string]$outDir = "C:\Work_Work\Machinery\report_figures\theory")
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force $outDir | Out-Null

$FF = New-Object System.Drawing.FontFamily('Leelawadee UI')
function F([float]$px, [switch]$bold) {
    $st = if ($bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    New-Object System.Drawing.Font($FF, $px, $st, [System.Drawing.GraphicsUnit]::Pixel)
}
function C($hex) { [System.Drawing.ColorTranslator]::FromHtml($hex) }
function B($hex) { New-Object System.Drawing.SolidBrush (C $hex) }

$INK = '#1F2937'; $SUB = '#4B5563'; $LINE = '#374151'
$BLUE = @('#E8F0FB', '#2F6DB5'); $ORANGE = @('#FDF0E3', '#C9711F'); $GREEN = @('#E6F4EA', '#2E8B57')

function NewCanvas([int]$w, [int]$h) {
    $bmp = New-Object System.Drawing.Bitmap($w, $h)
    $bmp.SetResolution(300, 300)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::White)
    return , @($bmp, $g)
}

function RRPath([float]$x, [float]$y, [float]$w, [float]$h, [float]$r) {
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = 2 * $r
    $p.AddArc($x, $y, $d, $d, 180, 90)
    $p.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $p.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
    $p.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $p.CloseFigure()
    return $p
}

# items: list of @(text, font, colorHex); drawn as one block centred in the rectangle
function TextBlock($g, $items, [float]$x, [float]$y, [float]$w, [float]$h, [string]$align = 'Center') {
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::$align
    $hs = @(); $total = 0
    foreach ($it in $items) { $s = $g.MeasureString($it[0], $it[1], [int]$w, $sf); $hs += $s.Height; $total += $s.Height }
    $cy = $y + ($h - $total) / 2
    for ($i = 0; $i -lt $items.Count; $i++) {
        $it = $items[$i]
        $rect = New-Object System.Drawing.RectangleF($x, $cy, $w, $hs[$i])
        $g.DrawString($it[0], $it[1], (B $it[2]), $rect, $sf)
        $cy += $hs[$i]
    }
}

function Box($g, [float]$x, [float]$y, [float]$w, [float]$h, $col, $items) {
    $path = RRPath $x $y $w $h 24
    $g.FillPath((B $col[0]), $path)
    $g.DrawPath((New-Object System.Drawing.Pen((C $col[1]), 5)), $path)
    TextBlock $g $items ($x + 20) ($y + 12) ($w - 40) ($h - 24)
}

function NumDot($g, [float]$cx, [float]$cy, [string]$n, $col) {
    $r = 32
    $g.FillEllipse((B $col[1]), $cx - $r, $cy - $r, 2 * $r, 2 * $r)
    $g.DrawEllipse((New-Object System.Drawing.Pen([System.Drawing.Color]::White, 4)), $cx - $r, $cy - $r, 2 * $r, 2 * $r)
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = 'Center'; $sf.LineAlignment = 'Center'
    $g.DrawString($n, (F 36 -bold), [System.Drawing.Brushes]::White, (New-Object System.Drawing.RectangleF(($cx - $r), ($cy - $r + 2), (2 * $r), (2 * $r))), $sf)
}

function Arrow($g, [float[]]$pts, [string]$color = $LINE, [switch]$dash) {
    $pen = New-Object System.Drawing.Pen((C $color), 6)
    $pen.CustomEndCap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(4, 5, $true)
    $pen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
    if ($dash) { $pen.DashPattern = [float[]]@(3, 2) }
    $p = New-Object 'System.Collections.Generic.List[System.Drawing.PointF]'
    for ($i = 0; $i -lt $pts.Count; $i += 2) { $p.Add((New-Object System.Drawing.PointF($pts[$i], $pts[$i + 1]))) }
    $g.DrawLines($pen, $p.ToArray())
}

function Label($g, [string]$text, $font, [string]$color, [float]$x, [float]$y, [float]$w, [float]$h, [string]$align = 'Center') {
    TextBlock $g (, @($text, $font, $color)) $x $y $w $h $align
}

$tF = F 42 -bold; $bF = F 34; $cF = F 30; $lF = F 36 -bold

# ---------------------------------------------------------------- figure 2.2
$cv = NewCanvas 1800 1000; $bmp = $cv[0]; $g = $cv[1]

Box $g 80 250 560 310 $BLUE @(
    , @('Agent: policy π', $tF, $BLUE[1])
    , @('โครงข่ายประสาทแบบ MLP', $bF, $INK)
    , @('256-128-128 (ELU)', $bF, $INK)
    , @('เข้า 56 ค่า / ออก 14 ค่า', $cF, $SUB))
Box $g 1160 250 560 310 $ORANGE @(
    , @('สภาพแวดล้อม', $tF, $ORANGE[1])
    , @('Isaac Sim / PhysX', $bF, $INK)
    , @('หุ่น 4,096 ตัวพร้อมกัน', $bF, $INK)
    , @('ฟิสิกส์ทุก 0.005 s, สั่งงาน 50 Hz', $cF, $SUB))
Box $g 620 760 560 200 $GREEN @(
    , @('PPO (rsl_rl)', $tF, $GREEN[1])
    , @('เก็บข้อมูล 24 ก้าว × 4,096 ตัว', $bF, $INK)
    , @('ปรับน้ำหนัก policy (3,000 รอบ)', $cF, $SUB))

Arrow $g @(360, 250, 360, 150, 1440, 150, 1440, 250)            # action
Arrow $g @(1340, 560, 1340, 670, 460, 670, 460, 560)            # state
Arrow $g @(1540, 560, 1540, 860, 1180, 860) $GREEN[1]           # reward -> PPO
Arrow $g @(620, 860, 260, 860, 260, 560) $GREEN[1]              # update policy

Label $g 'การกระทำ a(t): มุมเป้าหมายของข้อต่อ 14 ค่า' $lF $INK 380 80 1040 64
Label $g 'สถานะ s(t+1): 56 ค่า' $lF $INK 480 606 840 60
Label $g 'รางวัล r(t)' $lF $GREEN[1] 1190 872 340 54
Label $g '(เดินตามคำสั่ง ตัวตั้งตรง)' $cF $SUB 1190 922 360 50
Label $g 'ปรับ policy' $lF $GREEN[1] 270 800 340 56
Label $g 'วนซ้ำทุก 0.02 s' $cF $SUB 640 380 520 50

$bmp.Save((Join-Path $outDir 'fig_rl_loop.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
"saved fig_rl_loop.png"

# ---------------------------------------------------------------- figure 3.2
$cv = NewCanvas 1800 1110; $bmp = $cv[0]; $g = $cv[1]
$W = 460; $H = 260
$cx = @(60, 670, 1280); $ry = @(60, 430, 800)

$steps = @(
    @(0, 0, '1', $BLUE, @(@('URDF ของหุ่น', $tF, $BLUE[1]), @('ไฟล์โมเดลจากผู้ออกแบบ', $bF, $INK), @('ข้อต่อหมุน 15 ข้อ', $cF, $SUB))),
    @(1, 0, '2', $BLUE, @(@('แก้มวลและการชน', $tF, $BLUE[1]), @('fix_urdf.py', $cF, $SUB), @('มวลและ inertia จาก mesh', $bF, $INK), @('เพิ่มรูปทรงการชน', $bF, $INK))),
    @(2, 0, '3', $BLUE, @(@('หาท่ายืน', $tF, $BLUE[1]), @('find_stance.py', $cF, $SUB), @('ไล่มุม hip / knee / ankle', $bF, $INK), @('ให้ CoM อยู่กลางเท้า', $bF, $INK))),
    @(2, 1, '4', $ORANGE, @(@('ฝึกใน Isaac Lab', $tF, $ORANGE[1]), @('PPO, หุ่น 4,096 ตัว', $bF, $INK), @('3,000 รอบต่อครั้ง', $bF, $INK))),
    @(0, 1, '5', $ORANGE, @(@('ตรวจวิดีโอ', $tF, $ORANGE[1]), @('play / record_video', $cF, $SUB), @('เดินจริง หรือโกงรางวัล', $bF, $INK))),
    @(0, 2, '6', $GREEN, @(@('บันทึกท่าเดิน', $tF, $GREEN[1]), @('record_gait.py', $cF, $SUB), @('4 ความเร็ว 0.10-0.25 m/s', $bF, $INK), @('มุม แรงบิด ตำแหน่งเท้า', $cF, $INK))),
    @(1, 2, '7', $GREEN, @(@('วิเคราะห์กลไก', $tF, $GREEN[1]), @('kinematics_report.py', $cF, $SUB), @('limit_position.py', $cF, $SUB), @('DOF, Jacobian, workspace', $cF, $INK), @('limit position, แรงบิด', $cF, $INK)))
)
foreach ($s in $steps) {
    $x = $cx[$s[0]]; $y = $ry[$s[1]]
    Box $g $x $y $W $H $s[3] $s[4]
}

Arrow $g @(520, 185, 670, 185)                     # 1 -> 2
Arrow $g @(1130, 185, 1280, 185)                   # 2 -> 3
Arrow $g @(1510, 320, 1510, 430)                   # 3 -> 4
Arrow $g @(1280, 510, 520, 510)                    # 4 -> 5
Arrow $g @(520, 590, 1280, 590) $ORANGE[1] -dash   # 5 -> 4 (retrain)
Arrow $g @(290, 690, 290, 800)                     # 5 -> 6
Arrow $g @(520, 930, 670, 930)                     # 6 -> 7

Label $g 'policy ที่ฝึกเสร็จ' $lF $INK 530 452 740 52
Label $g 'พบ reward hacking: แก้ reward และเงื่อนไขจบ' $cF $ORANGE[1] 530 604 740 44
Label $g 'แล้วฝึกใหม่ (รอบที่ 1 ถึง 4)' $cF $ORANGE[1] 530 646 740 44
Label $g 'เดินได้จริง (รอบที่ 4)' $cF $INK 305 718 360 52 'Near'

foreach ($s in $steps) { NumDot $g ($cx[$s[0]] + 4) ($ry[$s[1]] + 4) $s[2] $s[3] }

# legend in the free cell
$lx = 1300; $ly = 830
$leg = @(@($BLUE, 'เตรียมโมเดลหุ่น'), @($ORANGE, 'ฝึกและตรวจผล'), @($GREEN, 'เก็บข้อมูลและวิเคราะห์'))
for ($i = 0; $i -lt 3; $i++) {
    $yy = $ly + $i * 72
    $p = RRPath $lx $yy 52 52 10
    $g.FillPath((B $leg[$i][0][0]), $p)
    $g.DrawPath((New-Object System.Drawing.Pen((C $leg[$i][0][1]), 5)), $p)
    Label $g $leg[$i][1] $bF $INK ($lx + 72) ($yy - 4) 380 60 'Near'
}

$bmp.Save((Join-Path $outDir 'fig_workflow.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
"saved fig_workflow.png"
