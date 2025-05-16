import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="Network Visualization",
    layout="wide"
)

# Custom CSS for rounded corners and removing gray lines
st.markdown("""
<style>
    /* Remove all borders and gray lines */
    iframe {
        border-radius: 15px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        border: none !important;
        outline: none !important;
    }
    
    /* Target Streamlit containers to remove gray lines */
    .css-1d391kg, .css-1kyxreq, .stApp, div[data-testid="stVerticalBlock"] {
        border: none !important;
        outline: none !important;
    }
    
    /* Target more specific Streamlit elements that might have borders */
    .element-container, .main, .block-container, .css-18e3th9 {
        border: none !important;
        outline: none !important;
    }
    
    /* Remove all gray outlines/borders from the visualization */
    .stApp {
        background-color: white;
    }
    
    /* Style the container to remove gray lines */
    .css-18e3th9 {
        padding-top: 0;
        padding-bottom: 0;
    }
    
    /* Target the html component container */
    [data-testid="stHtmlFrame"] {
        border: none !important;
        outline: none !important;
        background: white;
        border-radius: 15px;
    }
    
    /* Hide scrollbars */
    ::-webkit-scrollbar {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Add a title
st.title("Network Visualization")

# Function to read the HTML file and inject custom loading screen
def get_html_with_loading(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Add CSS to hide the original loading bar instead of removing it
    css_to_add = """
    <style>
    /* Hide the original loading bar but don't remove it */
    #loadingBar {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }
    
    /* Remove borders from card and visualization elements */
    .card, .card-body, #mynetwork {
        border: none !important;
        outline: none !important;
    }
    
    /* Override any existing borders */
    * {
        border-color: transparent;
    }
    
    /* Custom loading screen */
    #custom-loading {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.95);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #000;
        border-radius: 50%;
        animation: spin 1.5s linear infinite;
    }
    
    .loading-text {
        margin-top: 15px;
        font-size: 14px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #333;
    }
    
    .progress-bar-container {
        width: 200px;
        height: 2px;
        background-color: #f3f3f3;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        width: 0%;
        background-color: #000;
        transition: width 0.3s ease;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """
    
    # Find </head> tag and insert CSS just before it
    head_close_index = html_content.find("</head>")
    if head_close_index != -1:
        html_content = html_content[:head_close_index] + css_to_add + html_content[head_close_index:]
    
    # Add our custom loading screen
    loading_html = """
    <div id="custom-loading">
        <div class="spinner"></div>
        <div class="progress-bar-container">
            <div id="progress-bar" class="progress-bar"></div>
        </div>
        <div class="loading-text">Loading Network</div>
        <div id="loading-percent">0%</div>
    </div>
    
    <script>
    // Simple timer-based loading animation
    let progress = 0;
    const progressBar = document.getElementById('progress-bar');
    const loadingPercent = document.getElementById('loading-percent');
    const loadingElement = document.getElementById('custom-loading');
    
    function updateLoading() {
        if (progress < 100) {
            progress += Math.random() * 10;
            if (progress > 100) progress = 100;
            progressBar.style.width = progress + '%';
            loadingPercent.innerText = Math.round(progress) + '%';
            
            if (progress < 100) {
                setTimeout(updateLoading, 400);
            } else {
                setTimeout(() => {
                    loadingElement.style.display = 'none';
                }, 500);
            }
        }
    }
    
    // Start the loading animation
    document.addEventListener('DOMContentLoaded', function() {
        updateLoading();
    });
    </script>
    """
    
    # Find <body> tag and insert loading screen after it
    body_index = html_content.find("<body>")
    if body_index != -1:
        modified_html = html_content[:body_index + 6] + loading_html + html_content[body_index + 6:]
        return modified_html
    else:
        return html_content

# Function to render HTML in Streamlit without padding
def render_html(html_content):
    st.components.v1.html(html_content, height=650, scrolling=False)

# Main app function
def main():
    # Sidebar information (with reduced padding)
    st.sidebar.title("Visualization Controls")
    st.sidebar.info("This application displays a network visualization using vis.js.")
    
    # Create container with no padding
    container = st.container()
    
    # Path to HTML file
    html_file_path = "dynamic_kg.html"
    
    try:
        # Get HTML content with loading screen
        html_with_loading = get_html_with_loading(html_file_path)
        
        # Render the HTML content
        with container:
            render_html(html_with_loading)
        
        # Add explanation
        st.markdown("""
        ### About this Visualization
        
        This network visualization shows various nodes representing concepts and relationships. 
        The visualization is created using vis.js, a dynamic, browser-based visualization library.
        
        You can interact with the network by:
        - Dragging nodes to reposition them
        - Zooming in and out using the mouse wheel
        - Panning by dragging the background
        """)
    except FileNotFoundError:
        st.error(f"Error: The file '{html_file_path}' was not found. Please ensure the file exists in the same directory as this Streamlit application.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.code(str(e))  # Show detailed error message for debugging

if __name__ == "__main__":
    main()