import os

INPUT_DIR = "batch3a"
OUTPUT_DIR = "batch3a"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_tflite_to_header(tflite_path, header_path, var_name):

    with open(tflite_path, "rb") as f:
        data = f.read()

    # pecah 12 byte per baris
    lines = []
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append("  " + hex_values)

    hex_array = ",\n".join(lines)

    header_content = f"""
#ifndef {var_name.upper()}_H
#define {var_name.upper()}_H

#include <cstdint>

alignas(16) const unsigned char {var_name}[] = {{
{hex_array}
}};

const unsigned int {var_name}_len = {len(data)};

#endif
"""

    with open(header_path, "w") as f:
        f.write(header_content)


for file in os.listdir(INPUT_DIR):

    if file.endswith(".tflite"):

        model_name = file.replace(".tflite", "")

        tflite_path = os.path.join(INPUT_DIR, file)
        header_path = os.path.join(OUTPUT_DIR, model_name + ".h")

        convert_tflite_to_header(
            tflite_path,
            header_path,
            model_name
        )

        print(f"Converted: {file} -> {model_name}.h")

print("\nSelesai konversi semua model.")