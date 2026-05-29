import re

with open(r'c:\Users\ahedo\Documents\coding\Pymatgen\MetaForge-Web\templates\index.html', 'r', encoding='utf-8') as f:
    content = f.read()

script_start = content.find('<script>')
script_end = content.find('</script>') + len('</script>')

new_script = '''<script>
        // Define the 4 Metallurgical Universes
        const ALLOY_FAMILIES = {
            "Aerospace Alloy": [
                { sym: 'Al', name: 'Aluminium', color: '#3b82f6' },
                { sym: 'Ti', name: 'Titanium', color: '#8b5cf6' },
                { sym: 'Sc', name: 'Scandium', color: '#10b981' },
                { sym: 'Zr', name: 'Zirconium', color: '#f59e0b' },
                { sym: 'V', name: 'Vanadium', color: '#ef4444' }
            ],
            "Lightweight Alloy": [
                { sym: 'Al', name: 'Aluminium', color: '#3b82f6' },
                { sym: 'Mg', name: 'Magnesium', color: '#8b5cf6' },
                { sym: 'Li', name: 'Lithium', color: '#10b981' },
                { sym: 'Ti', name: 'Titanium', color: '#f59e0b' },
                { sym: 'Zn', name: 'Zinc', color: '#ef4444' }
            ],
            "Refractory Alloy": [
                { sym: 'W', name: 'Tungsten', color: '#3b82f6' },
                { sym: 'Mo', name: 'Molybdenum', color: '#8b5cf6' },
                { sym: 'Ta', name: 'Tantalum', color: '#10b981' },
                { sym: 'Nb', name: 'Niobium', color: '#f59e0b' },
                { sym: 'V', name: 'Vanadium', color: '#ef4444' }
            ],
            "Corrosion Resistance": [
                { sym: 'Co', name: 'Cobalt', color: '#3b82f6' },
                { sym: 'Cr', name: 'Chromium', color: '#8b5cf6' },
                { sym: 'Fe', name: 'Iron', color: '#10b981' },
                { sym: 'Ni', name: 'Nickel', color: '#f59e0b' },
                { sym: 'Cu', name: 'Copper', color: '#ef4444' }
            ]
        };

        let currentCategory = "Aerospace Alloy";
        let elements = ALLOY_FAMILIES[currentCategory];
        let state = {};
        let resultsComp = {};
        let debounceTimer = null;

        const container = document.getElementById('sliders-container');
        const donutChart = document.getElementById('donutChart');
        const donutLegend = document.getElementById('donutLegend');
        const loadingBar = document.getElementById('loadingBar');
        const tabsContainer = document.getElementById('categoryTabs');

        // Initialize state to 20% for each element
        function initState() {
            state = {};
            elements.forEach(el => state[el.sym] = 20);
        }

        function renderTabs() {
            if (!tabsContainer) return;
            tabsContainer.innerHTML = Object.keys(ALLOY_FAMILIES).map(cat => 
                <button class="tab-btn ""{cat === currentCategory ? 'active' : ''}" 
                        onclick="switchCategory('""{cat}')">
                    ""{cat}
                </button>
            ).join('');
        }

        function switchCategory(cat) {
            currentCategory = cat;
            elements = ALLOY_FAMILIES[cat];
            initState();
            renderTabs();
            initUI();
        }

        function initUI() {
            container.innerHTML = elements.map(el => 
                <div class="slider-group">
                    <div class="slider-header">
                        <div class="el-label">
                            <div class="el-dot" style="background-color: ""{el.color}"></div>
                            ""{el.sym} <span style="color: var(--text-muted); font-size: 0.85rem; font-weight: 400; margin-left: 4px;">""{el.name}</span>
                        </div>
                        <div class="input-group">
                            <input type="number" id="num-""{el.sym}" aria-label="""{el.name} percentage input" value="""{state[el.sym]}" min="0" max="100">
                            <div class="normalized-val" id="norm-""{el.sym}">0.0%</div>
                        </div>
                    </div>
                    <input type="range" id="range-""{el.sym}" aria-label="""{el.name} slider" value="""{state[el.sym]}" min="0" max="100">
                </div>
            ).join('');

            elements.forEach(el => {
                const numInput = document.getElementById(
um-""{el.sym});
                const rangeInput = document.getElementById(ange-""{el.sym});

                const update = (val) => {
                    let parsed = parseInt(val, 10);
                    if (isNaN(parsed)) parsed = 0;
                    if (parsed < 0) parsed = 0;
                    if (parsed > 100) parsed = 100;
                    
                    state[el.sym] = parsed;
                    numInput.value = parsed;
                    rangeInput.value = parsed;
                    schedulePredict();
                };

                numInput.addEventListener('input', e => update(e.target.value));
                rangeInput.addEventListener('input', e => update(e.target.value));
            });

            predict(); // Initial call
        }

        function schedulePredict() {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(predict, 300);
        }

        function updateDonut() {
            let svgContent = '';
            let legendContent = '';
            let cumulative = 0;
            
            elements.forEach(el => {
                const pct = resultsComp[el.sym] || 0;
                if (pct > 0) {
                    const dasharray = ""{pct} 100;
                    const dashoffset = -cumulative;
                    svgContent += <circle r="15.91549430918954" cx="16" cy="16" fill="transparent" stroke="""{el.color}" stroke-width="6" stroke-dasharray="""{dasharray}" stroke-dashoffset="""{dashoffset}"></circle>;
                    cumulative += pct;

                    legendContent += 
                        <div class="legend-item">
                            <div class="el-dot" style="background-color: ""{el.color}"></div>
                            ""{el.sym} ""{pct.toFixed(1)}%
                        </div>
                    ;
                }
            });

            donutChart.innerHTML = svgContent || <circle r="15.91549430918954" cx="16" cy="16" fill="transparent" stroke="var(--border-subtle)" stroke-width="6"></circle>;
            donutLegend.innerHTML = legendContent;
        }

        async function predict() {
            loadingBar.style.opacity = '1';
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(state)
                });
                const data = await res.json();
                
                document.getElementById('val-density').innerText = data.density;
                document.getElementById('val-strength').innerText = data.strength;
                document.getElementById('val-score').innerText = data.score;
                
                resultsComp = data.composition;
                
                // Update normalized labels
                elements.forEach(el => {
                    const normEl = document.getElementById(
orm-""{el.sym});
                    if (normEl) {
                        normEl.innerText = (resultsComp[el.sym] || 0).toFixed(1) + '%';
                    }
                });

                updateDonut();
            } catch (err) {
                console.error(err);
            } finally {
                loadingBar.style.opacity = '0';
            }
        }

        // Boot sequence
        initState();
        renderTabs();
        initUI();
    </script>'''.replace('""', '$')

content = content[:script_start] + new_script + content[script_end:]

with open(r'c:\Users\ahedo\Documents\coding\Pymatgen\MetaForge-Web\templates\index.html', 'w', encoding='utf-8') as f:
    f.write(content)
