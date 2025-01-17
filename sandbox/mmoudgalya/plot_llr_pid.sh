#cd /exp/uboone/app/users/mmoudgal/PELEE/sandbox/mmoudgalya

# condasetup
# conda activate python3LEE
# echo "Activated python3LEE environment."
python 1-load_data_n_store.py
echo "Stage 1 completed."
echo " "
conda deactivate
echo "Back to base environment."
conda activate mimLEE
echo "Activated mimLEE environment."
python 2-explode.py
echo "Stage 2 completed."
echo " "
conda deactivate
echo "Back to base environment."
conda activate python3LEE
echo "Activated python3LEE environment."
python 3-plot_llrpid.py
echo "Stage 3 completed."