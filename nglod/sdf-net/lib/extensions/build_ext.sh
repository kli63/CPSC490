# Install C++/CUDA extensions
for ext in mesh2sdf_cuda sol_nglod; do
    if [ -d "$ext" ]; then
        echo "Building $ext..."
        cd $ext && python setup.py clean --all install && cd ..
    else
        echo "Directory $ext not found"
    fi
done
