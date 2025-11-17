{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mFailed to start the Kernel. \n",
      "\u001b[1;31mUnable to start Kernel 'env_311 (Python 3.11.11)' due to a timeout waiting for the ports to get used. \n",
      "\u001b[1;31mView Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details."
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load CSV\n",
    "df = pd.read_csv(\"reddit_data.csv\")\n",
    "\n",
    "# Number of unique users\n",
    "n_users = df[\"username\"].nunique()\n",
    "print(f\"Number of unique users: {n_users}\")\n",
    "\n",
    "# Number of rows per user\n",
    "rows_per_user = df[\"username\"].value_counts()\n",
    "\n",
    "print(\"\\nRows per user:\")\n",
    "print(rows_per_user)\n",
    "\n",
    "# Summary statistics\n",
    "print(\"\\nSummary statistics (rows per user):\")\n",
    "print(rows_per_user.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "env_311",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
