#!/usr/bin/env bash
# A quick and dirty conversion of exposed f1r3node prometheus metric counters to grafana json dashboard

header=$(cat <<-EOM
{
  "__inputs": [
    {
      "name": "DS_PROMETHEUS",
      "label": "Prometheus",
      "description": "",
      "type": "datasource",
      "pluginId": "prometheus",
      "pluginName": "Prometheus"
    }
  ],
  "__requires": [
    {
      "type": "grafana",
      "id": "grafana",
      "name": "Grafana",
      "version": "5.2.1"
    },
    {
      "type": "panel",
      "id": "graph",
      "name": "Graph",
      "version": "5.2.1"
    },
    {
      "type": "datasource",
      "id": "prometheus",
      "name": "Prometheus",
      "version": "5.2.1"
    }
  ],
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "panels": [
EOM
)

footer=$(cat <<-EOM
 ],
  "refresh": "30s",
  "schemaVersion": 16,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {
    "refresh_intervals": [
      "5s",
      "10s",
      "30s",
      "1m",
      "5m",
      "15m",
      "30m",
      "1h",
      "2h",
      "1d"
    ],
    "time_options": [
      "5m",
      "15m",
      "1h",
      "6h",
      "12h",
      "24h",
      "2d",
      "7d",
      "30d"
    ]
  },
  "timezone": "",
  "title": "F1r3Node Counter Metrics(All)",
  "uid": "f1r3node-metrics-all",
  "version": 3
}
EOM
)

#!/usr/bin/env bash

# Start
SOURCE_URL=${1:-"http://127.0.0.1:40403/metrics"}

# Read metrics text from URL only
metrics_text=$(curl -fsS "${SOURCE_URL}" 2>/dev/null || true)

# Validate that we have any content
if [ -z "${metrics_text}" ]; then
  echo "${header}"
  echo "  ]\n,\n  \"title\": \"F1r3Node Counter Metrics(All)\"" > /dev/null
  echo "${footer}"
  exit 0
fi

# Extract metric names:
# - drop comments
# - keep only lines with exactly two whitespace-separated fields (name and value)
# - strip any label set `{...}` if present
# - drop duplicates and sort
metric_names=$(echo "${metrics_text}" \
  | grep -v '^#' \
  | awk 'NF==2 {print $1}' \
  | sed 's/{.*}//' \
  | sort -u)

panels_text=""
count=0
echo "${header}"
for metric_name in ${metric_names}; do
    count=$((count+1))
    idx=$((count-1))
    col=$((idx % 2))
    row=$((idx / 2))
    x=$((col*12))
    y=$((row*9))
    panel_text=$(cat <<-EOM
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fill": 1,
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": ${x},
        "y": ${y}
      },
      "id": ${count},
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "links": [],
      "nullPointMode": "null",
      "percentage": false,
      "pointradius": 5,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "${metric_name}",
          "format": "time_series",
          "intervalFactor": 1,
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeShift": null,
      "title": "${metric_name}",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ]
    },
EOM
)
    panels_text="$panels_text\n$panel_text"
done

# Trim the trailing comma from the last panel
panels_text="${panels_text::-1}"
echo -e "${panels_text}"
echo "${footer}"
