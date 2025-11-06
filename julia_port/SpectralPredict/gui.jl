"""
    SpectralPredict GUI

Web-based graphical interface for SpectralPredict.jl

Run this file to start the GUI server:
    julia --project=. gui.jl

Then open your browser to: http://localhost:8080
"""

using HTTP
using JSON
using DataFrames
using CSV
using Dates

# Load SpectralPredict
include("src/SpectralPredict.jl")
using .SpectralPredict

const PORT = 8080
const DATA_CACHE = Dict{String, Any}()

# HTML Interface
const HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpectralPredict.jl - GUI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .content { padding: 40px; }
        .section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            border: 2px solid #e9ecef;
        }
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }
        input[type="text"], select {
            width: 100%;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s;
        }
        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .checkbox-group {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .checkbox-item {
            display: flex;
            align-items: center;
            padding: 10px;
            background: white;
            border-radius: 6px;
            border: 2px solid #dee2e6;
            cursor: pointer;
            transition: all 0.2s;
        }
        .checkbox-item:hover {
            border-color: #667eea;
            background: #f0f3ff;
        }
        .checkbox-item input {
            margin-right: 8px;
            cursor: pointer;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        #results {
            display: none;
            margin-top: 30px;
        }
        #status {
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-weight: 500;
            display: none;
        }
        .status-info { background: #d1ecf1; color: #0c5460; border: 2px solid #bee5eb; }
        .status-success { background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
        .status-error { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .results-table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        .results-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #dee2e6;
        }
        .results-table tr:hover {
            background: #f8f9fa;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .note {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            color: #856404;
        }
        .note strong { color: #000; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SpectralPredict.jl</h1>
            <p>Machine Learning for Spectral Analysis</p>
        </div>

        <div class="content">
            <form id="analysisForm">
                <!-- Data Input Section -->
                <div class="section">
                    <h2>1. Data Input</h2>
                    <div class="form-group">
                        <label for="spectraDir">Spectra Directory (full path):</label>
                        <input type="text" id="spectraDir" name="spectraDir"
                               placeholder="/Users/yourusername/data/spectra" required>
                    </div>
                    <div class="form-group">
                        <label for="referenceFile">Reference CSV File (full path):</label>
                        <input type="text" id="referenceFile" name="referenceFile"
                               placeholder="/Users/yourusername/data/reference.csv" required>
                    </div>
                    <div class="form-group">
                        <label for="idColumn">Sample ID Column Name:</label>
                        <input type="text" id="idColumn" name="idColumn"
                               placeholder="sample_id" required>
                    </div>
                    <div class="form-group">
                        <label for="targetColumn">Target Variable Column Name:</label>
                        <input type="text" id="targetColumn" name="targetColumn"
                               placeholder="protein_pct" required>
                    </div>
                </div>

                <!-- Models Section -->
                <div class="section">
                    <h2>2. Select Models</h2>
                    <div class="note">
                        <strong>Note:</strong> PLS is currently disabled due to implementation issues.
                        Ridge, Lasso, and NeuralBoosted work excellently for spectroscopy!
                    </div>
                    <label>Choose models to test:</label>
                    <div class="checkbox-group">
                        <label class="checkbox-item">
                            <input type="checkbox" name="models" value="Ridge" checked>
                            Ridge
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" name="models" value="Lasso">
                            Lasso
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" name="models" value="ElasticNet">
                            ElasticNet
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" name="models" value="RandomForest">
                            RandomForest
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" name="models" value="MLP">
                            MLP (Neural Net)
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" name="models" value="NeuralBoosted">
                            NeuralBoosted (Gradient Boosting)
                        </label>
                    </div>
                </div>

                <!-- Preprocessing Section -->
                <div class="section">
                    <h2>3. Preprocessing</h2>
                    <label>Select preprocessing methods:</label>
                    <div class="checkbox-group">
                        <label class="checkbox-item">
                            <input type="checkbox" name="preprocessing" value="raw">
                            Raw (no preprocessing)
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" name="preprocessing" value="snv" checked>
                            SNV
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" name="preprocessing" value="deriv">
                            Derivatives
                        </label>
                    </div>

                    <div class="form-group" style="margin-top: 20px;">
                        <label for="derivOrders">Derivative Orders (comma-separated, e.g., 1,2):</label>
                        <input type="text" id="derivOrders" name="derivOrders" value="1,2">
                    </div>
                </div>

                <!-- Advanced Options -->
                <div class="section">
                    <h2>4. Advanced Options</h2>
                    <div class="form-group">
                        <label for="nFolds">Cross-Validation Folds:</label>
                        <select id="nFolds" name="nFolds">
                            <option value="3">3 (fast)</option>
                            <option value="5" selected>5 (recommended)</option>
                            <option value="10">10 (thorough)</option>
                        </select>
                    </div>

                    <div class="checkbox-group">
                        <label class="checkbox-item">
                            <input type="checkbox" name="enableVariableSubsets" id="enableVariableSubsets">
                            Enable Variable Subsets (feature selection)
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" name="enableRegionSubsets" id="enableRegionSubsets">
                            Enable Region Subsets (spectral regions)
                        </label>
                    </div>
                </div>

                <!-- Run Analysis Button -->
                <button type="submit" class="btn" id="runBtn">
                    Run Analysis
                </button>

                <div id="status"></div>
            </form>

            <!-- Results Section -->
            <div id="results" class="section">
                <h2>Results</h2>
                <div id="resultsContent"></div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('analysisForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const btn = document.getElementById('runBtn');
            const status = document.getElementById('status');
            const results = document.getElementById('results');

            // Disable button and show loading
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Running Analysis...';
            status.className = 'status-info';
            status.style.display = 'block';
            status.textContent = 'Starting analysis... This may take several minutes.';
            results.style.display = 'none';

            // Collect form data
            const formData = new FormData(e.target);
            const data = {
                spectraDir: formData.get('spectraDir'),
                referenceFile: formData.get('referenceFile'),
                idColumn: formData.get('idColumn'),
                targetColumn: formData.get('targetColumn'),
                models: formData.getAll('models'),
                preprocessing: formData.getAll('preprocessing'),
                derivOrders: formData.get('derivOrders'),
                nFolds: parseInt(formData.get('nFolds')),
                enableVariableSubsets: formData.get('enableVariableSubsets') === 'on',
                enableRegionSubsets: formData.get('enableRegionSubsets') === 'on'
            };

            try {
                const response = await fetch('/run-analysis', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.success) {
                    status.className = 'status-success';
                    status.textContent = 'Analysis completed successfully! Tested ' + result.n_configs + ' configurations.';

                    // Display results
                    displayResults(result.results);
                    results.style.display = 'block';
                } else {
                    status.className = 'status-error';
                    status.textContent = 'Error: ' + result.error;
                }
            } catch (error) {
                status.className = 'status-error';
                status.textContent = 'Error: ' + error.message;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Analysis';
            }
        });

        function displayResults(results) {
            const container = document.getElementById('resultsContent');

            if (!results || results.length === 0) {
                container.innerHTML = '<p>No results to display.</p>';
                return;
            }

            let html = '<table class="results-table"><thead><tr>';
            html += '<th>Rank</th><th>Model</th><th>Preprocessing</th><th>R²</th><th>RMSE</th><th>MAE</th><th>Variables</th>';
            html += '</tr></thead><tbody>';

            results.slice(0, 20).forEach(row => {
                html += '<tr>';
                html += '<td>' + row.Rank + '</td>';
                html += '<td>' + row.Model + '</td>';
                html += '<td>' + row.Preprocessing + '</td>';
                html += '<td>' + row.R2.toFixed(4) + '</td>';
                html += '<td>' + row.RMSE.toFixed(4) + '</td>';
                html += '<td>' + row.MAE.toFixed(4) + '</td>';
                html += '<td>' + row.n_vars + '</td>';
                html += '</tr>';
            });

            html += '</tbody></table>';
            html += '<p style="margin-top: 20px; text-align: center;">Showing top 20 results</p>';

            container.innerHTML = html;
        }
    </script>
</body>
</html>
"""

# API endpoint to run analysis
function run_analysis(params::Dict)
    try
        println("Loading data...")

        # Parse derivative orders
        deriv_orders = parse.(Int, split(params["derivOrders"], ","))

        # Load data
        X, y, wavelengths, sample_ids = SpectralPredict.load_spectral_dataset(
            params["spectraDir"],
            params["referenceFile"],
            params["idColumn"],
            params["targetColumn"]
        )

        println("Data loaded: ", size(X, 1), " samples × ", size(X, 2), " wavelengths")
        println("Running search with ", length(params["models"]), " models...")

        # Run search
        results = SpectralPredict.run_search(
            X, y, wavelengths,
            task_type="regression",
            models=params["models"],
            preprocessing=params["preprocessing"],
            derivative_orders=deriv_orders,
            enable_variable_subsets=params["enableVariableSubsets"],
            enable_region_subsets=params["enableRegionSubsets"],
            n_folds=params["nFolds"]
        )

        println("Search complete! ", size(results, 1), " configurations tested")

        # Convert results to JSON-friendly format
        results_dict = []
        for row in eachrow(results)
            push!(results_dict, Dict(
                "Rank" => row.Rank,
                "Model" => row.Model,
                "Preprocessing" => row.Preprocessing,
                "R2" => row.R2,
                "RMSE" => row.RMSE,
                "MAE" => row.MAE,
                "n_vars" => row.n_vars
            ))
        end

        # Save results to file
        output_file = "spectralpredict_results_$(replace(string(now()), ":" => "-")).csv"
        SpectralPredict.save_results(results, output_file)
        println("Results saved to: ", output_file)

        return Dict(
            "success" => true,
            "results" => results_dict,
            "n_configs" => size(results, 1),
            "output_file" => output_file
        )

    catch e
        println("Error in analysis: ", e)
        println(stacktrace(catch_backtrace()))
        return Dict(
            "success" => false,
            "error" => string(e)
        )
    end
end

# HTTP request handler
function handle_request(req::HTTP.Request)
    # CORS headers
    headers = [
        "Content-Type" => "text/html; charset=utf-8",
        "Access-Control-Allow-Origin" => "*",
        "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type"
    ]

    # Route handling
    if req.method == "GET" && req.target == "/"
        return HTTP.Response(200, headers, HTML_PAGE)

    elseif req.method == "POST" && req.target == "/run-analysis"
        try
            # Parse JSON body
            params = JSON.parse(String(req.body))

            # Run analysis
            result = run_analysis(params)

            # Return JSON response
            headers[1] = "Content-Type" => "application/json"
            return HTTP.Response(200, headers, JSON.json(result))
        catch e
            headers[1] = "Content-Type" => "application/json"
            error_response = Dict("success" => false, "error" => string(e))
            return HTTP.Response(500, headers, JSON.json(error_response))
        end

    elseif req.method == "OPTIONS"
        return HTTP.Response(200, headers, "")

    else
        return HTTP.Response(404, headers, "Not Found")
    end
end

# Start server
function start_gui()
    println("=" ^ 80)
    println("Starting SpectralPredict GUI Server...")
    println("=" ^ 80)
    println()
    println("Server running at: http://localhost:$(PORT)")
    println()
    println("Open your web browser and navigate to: http://localhost:$(PORT)")
    println()
    println("Press Ctrl+C to stop the server")
    println("=" ^ 80)
    println()

    HTTP.serve(handle_request, "0.0.0.0", PORT)
end

# Run the server
if abspath(PROGRAM_FILE) == @__FILE__
    start_gui()
end
