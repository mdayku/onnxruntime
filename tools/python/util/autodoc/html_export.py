# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Interactive HTML Export for graph visualization.

Task 5.8: Creates standalone HTML files with embedded visualization
that can be opened in any browser without a server.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hierarchical_graph import HierarchicalGraph
    from .edge_analysis import EdgeAnalysisResult


# HTML template with embedded D3.js visualization
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - ONNX Autodoc</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-hover: #79c0ff;
            --border: #30363d;
            --success: #3fb950;
            --warning: #d29922;
            --error: #f85149;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            overflow: hidden;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
        }}
        
        .sidebar {{
            width: 320px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            padding: 20px;
            overflow-y: auto;
        }}
        
        .sidebar h1 {{
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: var(--accent);
        }}
        
        .sidebar h2 {{
            font-size: 1rem;
            color: var(--text-secondary);
            margin: 16px 0 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
        }}
        
        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--accent);
        }}
        
        .stat-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .controls {{
            margin-bottom: 20px;
        }}
        
        .btn {{
            display: inline-block;
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            cursor: pointer;
            font-size: 0.875rem;
            margin-right: 8px;
            margin-bottom: 8px;
            transition: all 0.2s;
        }}
        
        .btn:hover {{
            background: var(--accent);
            border-color: var(--accent);
            color: var(--bg-primary);
        }}
        
        .legend {{
            margin-top: 20px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
        }}
        
        .main {{
            flex: 1;
            position: relative;
        }}
        
        svg {{
            width: 100%;
            height: 100%;
            background: var(--bg-primary);
        }}
        
        .node {{
            cursor: pointer;
        }}
        
        .node:hover {{
            filter: brightness(1.2);
        }}
        
        .node-rect {{
            rx: 6;
            ry: 6;
        }}
        
        .node-label {{
            font-size: 11px;
            fill: white;
            text-anchor: middle;
            dominant-baseline: middle;
            pointer-events: none;
        }}
        
        .edge {{
            fill: none;
            stroke-linecap: round;
        }}
        
        .edge.bottleneck {{
            stroke: var(--error);
        }}
        
        .edge.attention {{
            stroke: var(--warning);
            stroke-dasharray: 5,3;
        }}
        
        .edge.skip {{
            stroke: var(--success);
            stroke-dasharray: 8,4;
        }}
        
        .tooltip {{
            position: absolute;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            font-size: 0.875rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 300px;
            z-index: 100;
        }}
        
        .tooltip.visible {{
            opacity: 1;
        }}
        
        .tooltip-title {{
            font-weight: bold;
            color: var(--accent);
            margin-bottom: 8px;
        }}
        
        .tooltip-row {{
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
        }}
        
        .tooltip-label {{
            color: var(--text-secondary);
        }}
        
        .minimap {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 200px;
            height: 150px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .minimap svg {{
            opacity: 0.6;
        }}
        
        .minimap-viewport {{
            fill: var(--accent);
            fill-opacity: 0.2;
            stroke: var(--accent);
            stroke-width: 2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <aside class="sidebar">
            <h1>{title}</h1>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="node-count">0</div>
                    <div class="stat-label">Nodes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="edge-count">0</div>
                    <div class="stat-label">Edges</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="peak-memory">0 MB</div>
                    <div class="stat-label">Peak Memory</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="depth">0</div>
                    <div class="stat-label">Depth</div>
                </div>
            </div>
            
            <h2>Controls</h2>
            <div class="controls">
                <button class="btn" onclick="expandAll()">Expand All</button>
                <button class="btn" onclick="collapseAll()">Collapse All</button>
                <button class="btn" onclick="resetZoom()">Reset View</button>
                <button class="btn" onclick="fitToScreen()">Fit to Screen</button>
            </div>
            
            <h2>Legend</h2>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #4A90D9;"></div>
                    <span>Convolution</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9B59B6;"></div>
                    <span>Linear/MatMul</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #E67E22;"></div>
                    <span>Attention</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #7F8C8D;"></div>
                    <span>Normalization</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #F1C40F;"></div>
                    <span>Activation</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2ECC71;"></div>
                    <span>Skip Connection</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #E74C3C;"></div>
                    <span>Memory Bottleneck</span>
                </div>
            </div>
        </aside>
        
        <main class="main">
            <svg id="graph"></svg>
            <div class="tooltip" id="tooltip"></div>
        </main>
    </div>
    
    <script>
        // Embedded graph data
        const graphData = {graph_json};
        const edgeData = {edge_json};
        
        // Category colors
        const categoryColors = {{
            'conv': '#4A90D9',
            'linear': '#9B59B6',
            'attention': '#E67E22',
            'norm': '#7F8C8D',
            'activation': '#F1C40F',
            'pool': '#1ABC9C',
            'embed': '#8E44AD',
            'reshape': '#3498DB',
            'elementwise': '#9B59B6',
            'reduce': '#E74C3C',
            'default': '#7F8C8D'
        }};
        
        // Get color for op type
        function getNodeColor(node) {{
            if (node.node_type === 'block') {{
                const blockType = (node.attributes?.block_type || '').toLowerCase();
                if (blockType.includes('attention')) return categoryColors.attention;
                if (blockType.includes('mlp') || blockType.includes('ffn')) return categoryColors.linear;
                if (blockType.includes('conv')) return categoryColors.conv;
                if (blockType.includes('norm')) return categoryColors.norm;
                if (blockType.includes('embed')) return categoryColors.embed;
                return categoryColors.default;
            }}
            
            const op = (node.op_type || '').toLowerCase();
            if (op.includes('conv')) return categoryColors.conv;
            if (op.includes('matmul') || op.includes('gemm')) return categoryColors.linear;
            if (op.includes('norm')) return categoryColors.norm;
            if (op.includes('relu') || op.includes('gelu') || op.includes('softmax')) return categoryColors.activation;
            if (op.includes('pool')) return categoryColors.pool;
            if (op.includes('reshape') || op.includes('transpose')) return categoryColors.reshape;
            if (op.includes('add') || op.includes('mul') || op.includes('sub')) return categoryColors.elementwise;
            if (op.includes('reduce')) return categoryColors.reduce;
            return categoryColors.default;
        }}
        
        // Initialize visualization
        const svg = d3.select('#graph');
        const container = svg.append('g');
        
        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {{
                container.attr('transform', event.transform);
            }});
        
        svg.call(zoom);
        
        // Tooltip
        const tooltip = d3.select('#tooltip');
        
        function showTooltip(event, node) {{
            const html = `
                <div class="tooltip-title">${{node.display_name || node.name}}</div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Type:</span>
                    <span>${{node.op_type || node.node_type}}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Nodes:</span>
                    <span>${{node.node_count || 1}}</span>
                </div>
                ${{node.total_flops > 0 ? `
                <div class="tooltip-row">
                    <span class="tooltip-label">FLOPs:</span>
                    <span>${{formatNumber(node.total_flops)}}</span>
                </div>
                ` : ''}}
                ${{node.total_memory_bytes > 0 ? `
                <div class="tooltip-row">
                    <span class="tooltip-label">Memory:</span>
                    <span>${{formatBytes(node.total_memory_bytes)}}</span>
                </div>
                ` : ''}}
            `;
            
            tooltip.html(html)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY + 10) + 'px')
                .classed('visible', true);
        }}
        
        function hideTooltip() {{
            tooltip.classed('visible', false);
        }}
        
        // Layout nodes
        function layoutNodes(nodes) {{
            const nodeWidth = 120;
            const nodeHeight = 40;
            const gapX = 40;
            const gapY = 60;
            
            let x = gapX;
            let y = gapY;
            let rowHeight = 0;
            
            const width = window.innerWidth - 360;
            
            nodes.forEach((node, i) => {{
                if (x + nodeWidth > width - gapX) {{
                    x = gapX;
                    y += rowHeight + gapY;
                    rowHeight = 0;
                }}
                
                node.x = x;
                node.y = y;
                node.width = nodeWidth;
                node.height = nodeHeight;
                
                x += nodeWidth + gapX;
                rowHeight = Math.max(rowHeight, nodeHeight);
            }});
            
            return nodes;
        }}
        
        // Render graph
        function render() {{
            container.selectAll('*').remove();
            
            // Flatten visible nodes
            const visibleNodes = [];
            function collectVisible(node) {{
                visibleNodes.push(node);
                if (!node.is_collapsed && node.children) {{
                    node.children.forEach(collectVisible);
                }}
            }}
            
            if (graphData.root) {{
                collectVisible(graphData.root);
            }}
            
            // Layout
            layoutNodes(visibleNodes);
            
            // Draw nodes
            const nodeGroups = container.selectAll('.node')
                .data(visibleNodes)
                .enter()
                .append('g')
                .attr('class', 'node')
                .attr('transform', d => `translate(${{d.x}}, ${{d.y}})`)
                .on('mouseover', showTooltip)
                .on('mouseout', hideTooltip)
                .on('click', (event, d) => {{
                    if (d.children && d.children.length > 0) {{
                        d.is_collapsed = !d.is_collapsed;
                        render();
                    }}
                }});
            
            nodeGroups.append('rect')
                .attr('class', 'node-rect')
                .attr('width', d => d.width)
                .attr('height', d => d.height)
                .attr('fill', d => getNodeColor(d))
                .attr('stroke', d => d.children && d.children.length > 0 ? 'white' : 'none')
                .attr('stroke-width', 2);
            
            nodeGroups.append('text')
                .attr('class', 'node-label')
                .attr('x', d => d.width / 2)
                .attr('y', d => d.height / 2)
                .text(d => truncate(d.display_name || d.name, 15));
            
            // Add expand indicator for parent nodes
            nodeGroups.filter(d => d.children && d.children.length > 0)
                .append('text')
                .attr('x', d => d.width - 8)
                .attr('y', 12)
                .attr('fill', 'white')
                .attr('font-size', '12px')
                .text(d => d.is_collapsed ? '+' : '-');
            
            // Update stats
            document.getElementById('node-count').textContent = visibleNodes.length;
            document.getElementById('edge-count').textContent = edgeData?.num_edges || 0;
            document.getElementById('peak-memory').textContent = formatBytes(edgeData?.peak_activation_bytes || 0);
            document.getElementById('depth').textContent = graphData.depth || 0;
        }}
        
        // Helper functions
        function truncate(str, len) {{
            return str.length > len ? str.slice(0, len) + '...' : str;
        }}
        
        function formatNumber(n) {{
            if (n >= 1e12) return (n / 1e12).toFixed(1) + 'T';
            if (n >= 1e9) return (n / 1e9).toFixed(1) + 'G';
            if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
            if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
            return n.toString();
        }}
        
        function formatBytes(bytes) {{
            if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
            if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
            if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
            return bytes + ' B';
        }}
        
        // Control functions
        function expandAll() {{
            function expand(node) {{
                node.is_collapsed = false;
                if (node.children) node.children.forEach(expand);
            }}
            if (graphData.root) expand(graphData.root);
            render();
        }}
        
        function collapseAll() {{
            function collapse(node) {{
                if (node.children && node.children.length > 0) {{
                    node.is_collapsed = true;
                }}
                if (node.children) node.children.forEach(collapse);
            }}
            if (graphData.root) collapse(graphData.root);
            render();
        }}
        
        function resetZoom() {{
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        }}
        
        function fitToScreen() {{
            const bounds = container.node().getBBox();
            const parent = svg.node().parentElement;
            const fullWidth = parent.clientWidth;
            const fullHeight = parent.clientHeight;
            const width = bounds.width;
            const height = bounds.height;
            const midX = bounds.x + width / 2;
            const midY = bounds.y + height / 2;
            
            const scale = 0.9 / Math.max(width / fullWidth, height / fullHeight);
            const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];
            
            svg.transition().duration(500).call(
                zoom.transform,
                d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
            );
        }}
        
        // Initial render
        render();
        setTimeout(fitToScreen, 100);
    </script>
</body>
</html>
"""


def generate_html(
    graph: "HierarchicalGraph",
    edge_result: "EdgeAnalysisResult | None" = None,
    title: str = "Model Architecture",
    output_path: Path | str | None = None,
) -> str:
    """
    Generate interactive HTML visualization.

    Task 5.8.1-5.8.5: Create standalone HTML with embedded visualization.

    Args:
        graph: HierarchicalGraph to visualize.
        edge_result: Optional edge analysis results.
        title: Page title.
        output_path: Optional path to save HTML file.

    Returns:
        HTML content as string.
    """
    # Convert graph to JSON
    graph_json = graph.to_json(indent=None)

    # Convert edge analysis to JSON
    if edge_result:
        edge_json = json.dumps(edge_result.to_dict())
    else:
        edge_json = "{}"

    # Generate HTML
    html = HTML_TEMPLATE.format(
        title=title,
        graph_json=graph_json,
        edge_json=edge_json,
    )

    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.write_text(html, encoding="utf-8")

    return html


class HTMLExporter:
    """
    Export ONNX models to interactive HTML visualization.

    Combines HierarchicalGraph and EdgeAnalysis into a single
    standalone HTML file.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("autodoc.html")

    def export(
        self,
        graph: "HierarchicalGraph",
        edge_result: "EdgeAnalysisResult | None" = None,
        output_path: Path | str = "model_graph.html",
        title: str | None = None,
    ) -> Path:
        """
        Export graph to HTML file.

        Args:
            graph: HierarchicalGraph to export.
            edge_result: Optional EdgeAnalysisResult for edge data.
            output_path: Output file path.
            title: Optional page title.

        Returns:
            Path to generated HTML file.
        """
        output_path = Path(output_path)

        if title is None:
            title = graph.root.name if graph.root else "Model"

        generate_html(
            graph=graph,
            edge_result=edge_result,
            title=title,
            output_path=output_path,
        )

        self.logger.info(f"HTML visualization exported to {output_path}")
        return output_path
