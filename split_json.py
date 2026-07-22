import os
import json

chunks_dir = os.path.join("storage", "app", "medical_processed_chunks")
max_items_per_file = 2000  # تقسيم الملف إلى أجزاء تحتوي كل منها على 2000 قطعة (حجم صغير وآمن جداً لـ GitHub)

if os.path.exists(chunks_dir):
    for filename in os.listdir(chunks_dir):
        if filename.endswith(".json") and not "_part_" in filename:
            file_path = os.path.join(chunks_dir, filename)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # إذا كان الملف كبيراً (أكبر من 30 ميجابايت) سنقوم بتقسيمه
            if file_size_mb > 30:
                print(f"جاري تقسيم الملف الكبير: {filename} (الحجم: {file_size_mb:.2f} MB)...")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                base_name = filename.replace(".json", "")
                for i in range(0, len(data), max_items_per_file):
                    chunk_data = data[i:i + max_items_per_file]
                    part_filename = f"{base_name}_part_{i // max_items_per_file + 1}.json"
                    part_path = os.path.join(chunks_dir, part_filename)
                    
                    with open(part_path, "w", encoding="utf-8") as pf:
                        json.dump(chunk_data, pf, ensure_ascii=False)
                
                # حذف الملف الضخم الأصلي الذي كان يمنع الرفع
                os.remove(file_path)
                print(f"تم تقسيم {filename} بنجاح وحذف النسخة الضخمة.")
    print("اكتملت عملية تجهيز وتصغير الملفات!")
else:
    print("مجلد الـ chunks غير موجود!")