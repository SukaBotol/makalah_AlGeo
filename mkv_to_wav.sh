echo "Converting MKV files to WAV..."
echo "================================"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found. Install it with:"
    echo "  sudo apt update && sudo apt install ffmpeg"
    exit 1
fi

# Find all .mkv files and convert
count=0
for file in *.mkv; do
    if [ -f "$file" ]; then
        # Get filename without extension
        filename="${file%.*}"
        output="${filename}.wav"
        
        echo "Converting: $file → $output"
        
        # Convert with 48kHz, mono, PCM
        ffmpeg -i "$file" -ac 1 -ar 48000 -q:a 0 "$output" -y 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Success"
            ((count++))
        else
            echo "  ✗ Failed"
        fi
    fi
done

echo "================================"
echo "Converted $count files"