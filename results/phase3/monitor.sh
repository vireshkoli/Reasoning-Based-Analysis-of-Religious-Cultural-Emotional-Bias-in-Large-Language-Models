#!/bin/bash
# Monitor Phase 3 pipeline progress every 5 minutes
LOG="/Users/vireshkoli/Documents/MTech/NLP Project/LLM/results/phase3/monitor_log.txt"
ABLATION_DIR="/Users/vireshkoli/Documents/MTech/NLP Project/LLM/results/phase3/ablation"
PLOTS_DIR="/Users/vireshkoli/Documents/MTech/NLP Project/LLM/results/phase3/plots"
METRICS_DIR="/Users/vireshkoli/Documents/MTech/NLP Project/LLM/results/phase3/metrics"
SARVAM_PID=28221

# Sarvam started at 04:08:35 IST Apr 11 (layer ablation)
# Head ablation started ~00:49 IST Apr 12
HEAD_START_EPOCH=$(date -j -f "%a %b %d %T %Y" "Sat Apr 12 00:49:00 2026" "+%s" 2>/dev/null || echo 0)

echo "$(date): Monitor started (PID $$)" | tee -a "$LOG"

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    NOW_EPOCH=$(date +%s)

    if kill -0 $SARVAM_PID 2>/dev/null; then
        ELAPSED=$(ps -p $SARVAM_PID -o etime= 2>/dev/null | tr -d ' ')
        CPU=$(ps -p $SARVAM_PID -o pcpu= 2>/dev/null | tr -d ' ')

        # Determine phase: layer done (file exists), so must be head ablation
        if [ -f "$ABLATION_DIR/layer_ablation_sarvam.json" ]; then
            # Head ablation: 5 layers x 16 heads x ~140s/head = ~11200s total (~3.1h)
            if [ "$HEAD_START_EPOCH" -gt 0 ]; then
                HEAD_ELAPSED=$((NOW_EPOCH - HEAD_START_EPOCH))
                HEAD_TOTAL=11200
                HEAD_PCT=$(( HEAD_ELAPSED * 100 / HEAD_TOTAL ))
                [ $HEAD_PCT -gt 100 ] && HEAD_PCT=99
                HEADS_DONE=$(( HEAD_ELAPSED / 140 ))
                HEADS_TOTAL=80
                [ $HEADS_DONE -gt $HEADS_TOTAL ] && HEADS_DONE=$HEADS_TOTAL
                REMAIN_SEC=$(( HEAD_TOTAL - HEAD_ELAPSED ))
                [ $REMAIN_SEC -lt 0 ] && REMAIN_SEC=0
                REMAIN_MIN=$(( REMAIN_SEC / 60 ))
                ETA=$(date -v +${REMAIN_MIN}M '+%H:%M IST' 2>/dev/null || echo "~${REMAIN_MIN}min")
                echo "$TS [SARVAM HEAD ABLATION] elapsed=$ELAPSED cpu=${CPU}% | est ~${HEADS_DONE}/80 heads (${HEAD_PCT}%) | ~${REMAIN_MIN}min remaining | ETA $ETA" | tee -a "$LOG"
            else
                echo "$TS [SARVAM HEAD ABLATION] elapsed=$ELAPSED cpu=${CPU}%" | tee -a "$LOG"
            fi
        else
            # Still on layer ablation
            ELAPSED_TOTAL=$((NOW_EPOCH - $(date -j -f "%a %b %d %T %Y" "Sat Apr 11 04:08:35 2026" "+%s" 2>/dev/null || echo NOW_EPOCH)))
            AVG_LAYER_SEC=2040
            EST_LAYER=$(( ELAPSED_TOTAL / AVG_LAYER_SEC ))
            [ $EST_LAYER -gt 28 ] && EST_LAYER=28
            echo "$TS [SARVAM LAYER ABLATION] elapsed=$ELAPSED cpu=${CPU}% | est ~${EST_LAYER}/28 layers" | tee -a "$LOG"
        fi
    else
        echo "$TS [SARVAM PROCESS DONE]" | tee -a "$LOG"

        # What's running next?
        ACTIVE=$(ps aux | grep -E "python.*phase3|python.*run_phase3|statistical|analyze|visualize" | grep -v grep | awk '{print $11, $12, $13}')
        [ -n "$ACTIVE" ] && echo "$TS [ACTIVE] $ACTIVE" | tee -a "$LOG"
    fi

    # Always show ablation files
    ABLATION_FILES=$(ls "$ABLATION_DIR"/*.json 2>/dev/null | xargs -I{} basename {} | tr '\n' ' ')
    echo "$TS [ABLATION FILES] ${ABLATION_FILES:-none}" | tee -a "$LOG"

    # Show plots once they appear
    PLOT_COUNT=$(ls "$PLOTS_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PLOT_COUNT" -gt 0 ]; then
        PLOT_LIST=$(ls "$PLOTS_DIR"/*.png 2>/dev/null | xargs -I{} basename {} | tr '\n' ' ')
        echo "$TS [PLOTS] ${PLOT_COUNT} files: $PLOT_LIST" | tee -a "$LOG"
        echo "$TS [PIPELINE COMPLETE] All done!" | tee -a "$LOG"
        exit 0
    fi

    echo "---" >> "$LOG"
    sleep 300
done
