import numpy as np
import matplotlib.pyplot as plt
import time
from operationclass import BranchSales

FILE_PATH = 'sales.txt'
BRANCHES = ['North', 'South', 'East', 'West', 'Central']

# ARRAY CREATION
def load_data():
    """Creating numpy arrays from file"""
    data = []
    with open(FILE_PATH, 'r') as f:
        for line in f:
            numbers = line.strip().split(',')
            nums = [int(x) for x in numbers if x]
            data.append(np.array(nums))
    return np.array(data, dtype='object')

# INDEXING & SLICING
def get_sales(data, branch, month=None):
    """Basic indexing into arrays"""
    if month is not None:
        return data[branch][month]
    return data[branch]

def top_months(data, branch, n=3):
    """Slicing and fancy indexing"""
    sales = data[branch]
    sorted_idx = np.argsort(sales)[::-1]
    return sales[sorted_idx[:n]]

# MATH & STATISTICAL OPERATIONS
def branch_total(data, branch):
    return np.sum(data[branch])

def best_and_worst(data):
    """Find best and worst using statistical ops"""
    totals = np.array([branch_total(data, i) for i in range(len(data))])
    best_idx = np.argmax(totals)
    worst_idx = np.argmin(totals)
    print(f"Best: {BRANCHES[best_idx]} - ${totals[best_idx]:,}")
    print(f"Worst: {BRANCHES[worst_idx]} - ${totals[worst_idx]:,}")

def all_stats(data):
    """Demonstrating mean, std, min, max"""
    print("\n--- Per Branch Monthly Averages ---")
    for i in range(len(data)):
        print(f"{BRANCHES[i]}: ${np.mean(data[i]):.2f}")

    all_sales = np.concatenate(data)
    print(f"\n--- Company Statistics ---")
    print(f"Total: ${np.sum(all_sales):,}")
    print(f"Mean: ${np.mean(all_sales):.2f}")
    print(f"Std Dev: ${np.std(all_sales):.2f}")
    print(f"Max: ${np.max(all_sales):,}")
    print(f"Min: ${np.min(all_sales):,}")

# AXIS-WISE OPERATIONS
def axis_operations(data):
    """Show axis-wise sums and means"""
    print("\n--- Axis-wise Operations (first 6 months) ---")
    matrix = np.array([data[i][:6] for i in range(len(data))])

    print(f"Shape: {matrix.shape}")
    print(f"Sum per month (axis=0): {np.sum(matrix, axis=0)}")
    print(f"Sum per branch (axis=1): {np.sum(matrix, axis=1)}")
    print(f"Mean per month: {np.mean(matrix, axis=0).round(1)}")
    print(f"Mean per branch: {np.mean(matrix, axis=1).round(1)}")

# RESHAPING
def quarterly_reshape(data):
    """Demonstrating reshape operation"""
    print("\n--- Reshaping into Quarters ---")
    all_sales = np.concatenate(data)
    print(f"Original shape: {all_sales.shape}")

    trim_to = len(all_sales) - (len(all_sales) % 3)
    trimmed = all_sales[:trim_to]
    print(f"Trimmed shape: {trimmed.shape}")

    quarters = trimmed.reshape(-1, 3)
    print(f"Reshaped to quarters: {quarters.shape}")

    for i, q in enumerate(quarters, 1):
        print(f"Q{i}: ${np.sum(q):,}")

# BROADCASTING
def broadcast_discounts(data):
    """Demonstrate broadcasting with different discount rates"""
    print("\n--- Broadcasting Example: Discounts ---")
    sales_matrix = np.array([data[i][:6] for i in range(len(data))])
    print(f"Sales matrix shape: {sales_matrix.shape}")

    discount_rates = np.array([0.05, 0.08, 0.03, 0.06, 0.10])
    print(f"Discount rates shape: {discount_rates.shape}")

    rates_reshaped = discount_rates[:, np.newaxis]
    print(f"Reshaped rates: {rates_reshaped.shape}")

    after_discount = sales_matrix * (1 - rates_reshaped)

    print("\nOriginal (first 6 months):")
    for i in range(len(BRANCHES)):
        print(f"{BRANCHES[i]}: {sales_matrix[i]}")

    print("\nAfter discount:")
    for i in range(len(BRANCHES)):
        print(f"{BRANCHES[i]}: {after_discount[i].astype(int)}")

# SAVE & LOAD OPERATIONS
def save_report(data):
    """Saving numpy arrays to files"""
    all_sales = np.concatenate(data)
    totals = np.array([branch_total(data, i) for i in range(len(data))])

    np.save('all_sales.npy', all_sales)
    np.save('branch_totals.npy', totals)

    print(f"\n--- Saved ---")
    print(f"Saved all sales: {all_sales.shape}")
    print(f"Saved branch totals: {totals.shape}")

def load_report():
    """Loading numpy arrays from files"""
    try:
        all_sales = np.load('all_sales.npy')
        totals = np.load('branch_totals.npy')

        print(f"\n--- Loaded ---")
        print(f"Loaded {len(all_sales)} sales records")
        print(f"Branch totals: {totals}")
        print(f"Grand total: ${np.sum(totals):,}")
        return all_sales, totals
    except:
        print("No saved files found")
        return None, None

# PERFORMANCE COMPARISON
def compare_performance():
    """Compare NumPy vs regular Python lists"""
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON: NumPy vs Python Lists")
    print("=" * 50)

    size = 1_000_000

    print(f"\nProcessing {size:,} transactions...")

    start = time.time()
    py_list = list(range(500, 500 + size))
    py_result = 0
    for x in py_list:
        py_result += (x * 0.9) - 50
    py_time = time.time() - start

    start = time.time()
    np_array = np.arange(500, 500 + size)
    np_result = np.sum((np_array * 0.9) - 50)
    np_time = time.time() - start

    print(f"\nPython List: {py_time:.4f} seconds")
    print(f"NumPy Array: {np_time:.4f} seconds")
    print(f"NumPy is {py_time / np_time:.1f}x faster!")
    print(f"\nResults match: ${py_result:,.0f} = ${np_result:,.0f}")

# CHARTS
def make_charts(data):
    """Create only 2 charts: all branches and totals"""
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    # Chart: All branches monthly sales
    plt.figure(figsize=(10, 6))
    for i in range(len(data)):
        plt.plot(data[i], 'o-', label=BRANCHES[i], color=colors[i], linewidth=2)
    plt.title('All Branches - Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_branches_sales.png')
    plt.show()

    # Chart: Total sales by branch
    plt.figure(figsize=(8, 5))
    totals = [branch_total(data, i) for i in range(len(data))]
    bars = plt.bar(BRANCHES, totals, color=colors)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'${bar.get_height():,}', ha='center', fontsize=9)
    plt.title('Total Sales by Branch')
    plt.ylabel('Total Sales ($)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('total_sales_by_branch.png')
    plt.show()


# MAIN
def main():
    print("=" * 30)
    print("NUMPY DATA EXPLORER - PROJECT 1")
    print("=" * 30)

    print("\nARRAY CREATION")
    print("-" * 30)
    sales = load_data()
    for i in range(len(sales)):
        print(f"{BRANCHES[i]}: {sales[i]}")

    print("\nBRANCH DETAILS")
    print("-" * 30)
    for i in range(len(sales)):
        branch = BranchSales(sales[i], BRANCHES[i])
        branch.show()

    print("\nINDEXING & SLICING")
    print("-" * 30)
    print(f"East Branch, month 2: ${get_sales(sales, 2, 1):,}")
    print(f"North Branch all months: {get_sales(sales, 0)}")
    print(f"North Branch top 3 months: {top_months(sales, 0)}")
    print(f"South Branch top 3: {top_months(sales, 1)}")

    print("\nMATH & STATISTICAL OPERATIONS")
    print("-" * 30)
    best_and_worst(sales)
    all_stats(sales)

    print("\nAXIS-WISE OPERATIONS")
    print("-" * 30)
    axis_operations(sales)

    print("\nRESHAPING")
    print("-" * 30)
    quarterly_reshape(sales)

    print("\nBROADCASTING")
    print("-" * 30)
    broadcast_discounts(sales)

    print("\nSAVE & LOAD OPERATIONS")
    print("-" * 30)
    save_report(sales)
    load_report()

    compare_performance()

    print("\nVISUALIZATIONS")
    print("-" * 30)
    make_charts(sales)

    print("\n" + "-" * 10)
    print("Done!")
    print("-" * 10)

if __name__ == "__main__":
    main()