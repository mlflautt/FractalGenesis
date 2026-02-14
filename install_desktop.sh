#!/bin/bash
# FractalGenesis Desktop Installation Script
# Adds the app to application menu and taskbar

echo "🌀 Installing FractalGenesis Desktop App..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Update the .desktop file with correct path
cat > ~/.local/share/applications/fractalgenesis.desktop << EOF
[Desktop Entry]
Name=FractalGenesis
Comment=Explore and evolve fractal parameter spaces
Exec=python3 ${SCRIPT_DIR}/fractalgenesis_gui.py
Icon=${SCRIPT_DIR}/assets/icon.png
Terminal=false
Type=Application
Categories=Graphics;Science;Art;
Keywords=fractal;evolution;genetic;algorithm;art;graphics;
StartupNotify=true
EOF

# Make it executable
chmod +x ~/.local/share/applications/fractalgenesis.desktop

# Update desktop database
update-desktop-database ~/.local/share/applications/

echo "✅ Added to Application Menu"

# Check if we're using GNOME and add to favorites
if command -v gsettings &> /dev/null; then
    echo "📝 Adding to GNOME favorites..."
    CURRENT_FAVORITES=$(gsettings get org.gnome.shell favorite-apps)
    
    # Check if already in favorites
    if [[ $CURRENT_FAVORITES == *"fractalgenesis.desktop"* ]]; then
        echo "ℹ️  Already in favorites"
    else
        # Add to favorites
        NEW_FAVORITES=$(echo $CURRENT_FAVORITES | sed "s/]$/, 'fractalgenesis.desktop']/")
        gsettings set org.gnome.shell favorite-apps "$NEW_FAVORITES"
        echo "✅ Added to taskbar/favorites"
    fi
fi

# Check if we're using KDE
if command -v qdbus &> /dev/null && pgrep plasmashell > /dev/null; then
    echo "📝 KDE Plasma detected - adding to taskbar..."
    # KDE taskbar configuration is more complex, would need kwriteconfig5
    echo "ℹ️  Please right-click on the app in the menu and select 'Add to Panel'"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "You can now:"
echo "  1. Find FractalGenesis in your application menu"
echo "  2. Launch it from the command line: python3 fractalgenesis_gui.py"
echo "  3. Right-click the .desktop file and select 'Add to Favorites'"
echo ""
echo "To launch now, run:"
echo "  python3 ${SCRIPT_DIR}/fractalgenesis_gui.py"
