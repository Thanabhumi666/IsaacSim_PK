ข้อมูลประกอบรายงาน Micro Duck (ดึงมาจากเครื่องอาจารย์ C:\Users\punn_\microduck_walk เมื่อ 18 พ.ย. 2569)

runs\     โฟลเดอร์ผลลัพธ์การฝึกฉบับเต็ม (คัดลอกทั้งโฟลเดอร์จาก logs\rsl_rl\microduck_flat)
  2026-11-05_16-48-38            รอบที่ 1
  2026-11-05_17-46-56_stdfix     รอบที่ 2
  2026-11-05_20-00-28_bipedfix   รอบที่ 3
  2026-11-06_17-34-35_upright    รอบที่ 4 (รอบที่เดินได้)
  2026-11-05_19-51-04_smoke, 2026-11-06_17-31-28_smoke2   รอบทดสอบสั้น ๆ ก่อนฝึกจริง ไม่ได้ใช้ในรายงาน
  ในแต่ละโฟลเดอร์รอบหลัก:
    model_0.pt ... model_2999.pt   policy ที่เซฟทุก 100 iteration (31 ไฟล์) ตัวสุดท้ายคือ model_2999.pt
    exported\policy.onnx, policy.pt   policy สุดท้ายที่ export แล้ว ใช้นำไปรันนอก Isaac Lab ได้
    events.out.tfevents.*          log ของ TensorBoard (ที่มาของกราฟการฝึกและตารางสรุปการฝึก)
    params\env.yaml, agent.yaml    ค่าตั้งของสภาพแวดล้อมและ PPO ที่ใช้ฝึกจริง
    git\*.diff                     โค้ดที่แก้จากต้นฉบับ Isaac Lab ตอนเริ่มฝึก
    videos\play\*.mp4              วิดีโอหุ่นเดิน

videos\   วิดีโอหุ่นเดินของรอบ 1-4 (สำเนาเดียวกับใน runs\ รวมไว้ให้เปิดดูง่าย, สั่ง vx = 0.2 m/s)

data\     ข้อมูลที่ใช้สร้างตารางและกราฟในรายงาน
  gait_*.npz / gait_*.csv   ข้อมูลท่าเดินจาก record_gait.py (มุม ความเร็ว แรงบิดข้อต่อ ตำแหน่งเท้า แรงที่เท้า)
                            gait_run4_0p10 ถึง 0p25 = รอบที่ 4 ที่ 4 ความเร็ว, gait_run1_final / gait_0p20 / gait_run3_final = รอบ 1-3
  *_result.txt              ผลการคำนวณ (จลนศาสตร์ ตำแหน่งสุดขีด การตรวจแรงที่เท้า ท่ายืน)
  compare_summary.txt       ตารางเปรียบเทียบท่าเดิน
  train_log.txt, train2_log.txt, train3_log.txt, train4_log.txt   ข้อความที่พิมพ์ออกมาระหว่างฝึกรอบที่ 1-4
  run_folders_manifest.txt  รายชื่อไฟล์ทั้งหมดในโฟลเดอร์ผลลัพธ์บนเครื่องอาจารย์
  *_log.txt อื่น ๆ          ข้อความจากการรันสคริปต์วิเคราะห์

assets\   โมเดลหุ่น
  microduck_fixed.urdf             URDF ที่แก้มวล ความเฉื่อย และรูปทรงการชนแล้ว (ใช้ฝึกและคำนวณ)
  microduck_description-main\      URDF ต้นฉบับและไฟล์รูปทรง 3D (meshes\*.stl) ที่ URDF อ้างถึง
  หมายเหตุ: microduck_fixed.urdf อ้างไฟล์ mesh ด้วยที่อยู่เต็มบนเครื่องอาจารย์
  (C:/Users/punn_/OneDrive/Desktop/IsaacSim_PK/microduck_description-main/meshes/...)
  ถ้าจะเปิดบนเครื่องอื่น ต้องแก้ที่อยู่นี้ให้ชี้ไปที่ assets\microduck_description-main\meshes

report_figures\   รูปและไฟล์ผลลัพธ์ทั้งหมดที่ใช้ในรายงาน

plots\    กราฟอื่น ๆ ที่สคริปต์วิเคราะห์สร้างไว้ (เส้นทางเท้า ความเร็วเท้า มุมข้อต่อ แรงบิด ของทุกความเร็ว และกราฟตำแหน่งสุดขีด)
          บางรูปไม่ได้ใช้ในรายงาน แต่เก็บไว้เป็นหลักฐานประกอบ

outputs\  บันทึกการรันแต่ละครั้งที่ Isaac Lab สร้างอัตโนมัติ (แยกตามวันและเวลา) มีค่าตั้งที่ใช้ (.yaml) และข้อความ log (.log)

MicroDuck_program_files.zip   โค้ดทั้งหมดตามภาคผนวก ก
