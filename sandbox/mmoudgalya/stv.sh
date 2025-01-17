cd /exp/uboone/app/users/mmoudgal/PELEE/sandbox/mmoudgalya

# condasetup
# conda activate python3LEE
# echo "Activated python3LEE environment."
python stv1_load_data_n_store.py
echo "Stage 1 completed."
echo " "
conda deactivate
echo "Back to base environment."
conda activate mimLEE
echo "Activated mimLEE environment."
python stv2_save_as_root.py
echo "Stage 2 completed."
echo " "
# conda deactivate
# echo "Back to base environment."
# conda activate python3LEE
# echo "Activated python3LEE environment."
