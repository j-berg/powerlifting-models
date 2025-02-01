import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

# Create a squat dataframe by selecting all columns where the first element of each column header tuple is 'Squat'
def parse_lift(data, lift_type='Squat'):
    lift_data = data[[col for col in data.columns if lift_type in col]]
    lift_data = lift_data.dropna()
    lift_data = lift_data.apply(pd.to_numeric)
    lift_data['Marker Size'] = lift_data[lift_type + '_Reps'].astype(int) * lift_data[lift_type + '_Sets'].astype(int) * 5
    lift_data['Date'] = lift_data.index
    return lift_data

# Function to plot rectangles with correct handling of dates
def plot_rectangles(ax, data, weight_col, date_col, reps_col, sets_col, color, label):
    for _, row in data.iterrows():
        date_num = mdates.date2num(row[date_col])  # Convert datetime to numerical format
        rect = Rectangle(
            (date_num, row[weight_col]),   # Bottom-left corner (date, weight)
            width=row[sets_col],           # Width (scaled by number of sets)
            height=row[reps_col],          # Height (scaled by number of reps)
            color=color, alpha=0.5, label=label
        )
        ax.add_patch(rect)
    ax.autoscale()  # Ensure the axes scale properly to fit the rectangles

class PowerliftingData(pd.DataFrame):
    # Usage:
    # df = PowerliftingData("path/to/your/data.csv")
    # # Optionally apply formatting:
    # df.format_weight_data()

    # Read in data on initialization, where the user provides the URL for the data
    def __init__(self, url, *args, **kwargs):
        # Load the data and initialize the DataFrame superclass with it
        data = pd.read_csv(url, *args, **kwargs)
        super().__init__(data)
        
        self._is_formatted = False

        # Optionally apply formatting if the user specifies
        self.format_weight_data()

    def format_weight_data(self):
        # Check if data formatting has already been applied
        if self._is_formatted:
            print("Data has already been formatted.")
            return self

        # Copy the original data
        data = self.copy()

        # Step 1: Forward-fill the first row to replace NaN values
        data.iloc[0] = data.iloc[0].fillna(method='ffill')

        # Step 2: Replace remaining NaNs in the first two rows with an empty string
        data.iloc[0] = data.iloc[0].fillna('')
        data.iloc[1] = data.iloc[1].fillna('')

        # Step 3: Combine first two rows to create multi-level headers
        data.columns = [f"{x}_{y}" if y else x for x, y in zip(data.iloc[0], data.iloc[1])]

        # Step 4: Drop the first two rows as they are now headers
        data = data.drop([0, 1]).reset_index(drop=True)

        # Step 5: Ensure 'Date' column values are filled down
        if 'Date' in data.columns:
            data['Date'] = data['Date'].fillna(method='ffill')
            data['Date'] = pd.to_datetime(data['Date'], format='mixed') #, errors='coerce')
            data = data.set_index('Date', drop=False)
        else:
            print("Date column not found.")
            return None
        
        data.index.name = None

        # For all Squat, Bench, and Deadlift columns, convert to numeric type
        for col in data.columns:
            if 'Squat' in col or 'Bench' in col or 'Deadlift' in col:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Step 6: Clean up "Unnamed" columns
        data.columns = [col.replace("Unnamed: ", "").strip() for col in data.columns]

        # Step 7: Properly update the DataFrame
        new_df = pd.DataFrame(data=data.values, 
                            columns=data.columns, 
                            index=data.index)
        self._update_inplace(new_df)
        
        # Mark formatting as completed
        self._is_formatted = True

        return None

    def split_by_lifts(self):
        
        # Check if data formatting has been applied
        if not self._is_formatted:
            print("Data has not been formatted. Please format the data first.")
            return self
        
        # Copy the original data
        data = self.copy()

        # Identify the lift types present 
        lift_types = set([col.split('_')[0] for col in self.columns if '_' in col])

        # Create a dictionary to store the dataframes for each lift type
        lift_data = {}
        for lift_type in lift_types:
            lift_data[lift_type] = parse_lift(data, lift_type)

        return lift_data
    
    def daily_max_weight_lift(self, lift_type, rep_filter=3):

        # Check if data formatting has been applied
        if not self._is_formatted:
            print("Data has not been formatted. Please format the data first.")
            return self

        # Identify the columns for the lift type
        weight_col = f"{lift_type}_Weight"
        reps_col = f"{lift_type}_Reps"

        # Copy the original data
        data = self.copy()

        # Filter for the lift type
        data = data[[col for col in data.columns if any(x in col for x in [lift_type, 'Date'])]]

        # Drop rows with Nan values row-wise 
        data = data.dropna(how='all') 

        if rep_filter is not None:
            data = data[data[reps_col] <= int(rep_filter)]

        # Get maximum weight for each date
        if len(data) > 0:
            idx_max_weight = data.groupby('Date')[weight_col].transform(max) == data[weight_col]
            return data[idx_max_weight].set_index('Date')
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data

    def filter_max_weights(self, rep_filter=3):
        """Filter data to show only the maximum weight lifted each day for each lift type."""
        
        lift_types = set([col.split('_')[0] for col in self.columns if '_' in col])

        lift_data = {}
        
        for lift_type in lift_types:
            max_weight_data = self.daily_max_weight_lift(lift_type, rep_filter=rep_filter)
            if not max_weight_data.empty:
                lift_data[lift_type] = max_weight_data
        
        return lift_data
    

class BodyCompositionData(pd.DataFrame):
    # Usage:
    # df = BodyCompositionData("path/to/your/data.csv")
    # # Optionally apply formatting:
    # df.format_body_composition_data()

    # Read in data on initialization, where the user provides the URL for the data
    def __init__(self, url, *args, **kwargs):
        # Load the data and initialize the DataFrame superclass with it
        data = pd.read_csv(url, *args, **kwargs)
        super().__init__(data)
        
        self._is_formatted = False

        # Optionally apply formatting if the user specifies
        self.format_body_composition_data()

    def format_body_composition_data(self):
        # Check if data formatting has already been applied
        if self._is_formatted:
            print("Data has already been formatted.")
            return self

        # Copy the original data
        data = self.copy()

        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        data.index.name = None

        #  Properly update the DataFrame
        new_df = pd.DataFrame(data=data.values, 
                            columns=data.columns, 
                            index=data.index)
        self._update_inplace(new_df)

        # Mark formatting as completed
        self._is_formatted = True


# Function to predict modified Epley 1RM
def predict_1rm(data, lift_type):
    return 0.0333 * data[lift_type+'_Weight'] * data[lift_type+'_Reps'] + data[lift_type+'_Weight']

# Function to perform weighted linear regression and extrapolate to the target goal
def predict_future_1rm(ax, data, lift_type, color, goal_weight):

    # Prepare X and y
    X = (data.index - data.index.min()).days.values.reshape(-1, 1)
    y = predict_1rm(data, lift_type).values

    # Perform initial linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calculate residuals (difference between actual and predicted values)
    residuals = y - y_pred

    # Assign weights - more weight to points where residual is positive (i.e., y > y_pred)
    weights = np.where(residuals > 0, 5.0, 0.5)  # Give higher weight to points above the line

    # Perform weighted linear regression
    model_weighted = LinearRegression()
    model_weighted.fit(X, y, sample_weight=weights)
    y_pred_weighted = model_weighted.predict(X)

    # Plot the weighted regression line
    ax.plot(data.index, y_pred_weighted, color=color, linestyle='dashed', alpha=0.7, label='Biased Trendline')

    # Plot daily 1RM predictions as points
    ax.scatter(data.index, y, color='gray', label='Predicted 1RM', s=3)

    # Extrapolate the regression to the goal weight
    slope = model_weighted.coef_[0]
    intercept = model_weighted.intercept_

    # Calculate the number of days required to reach the goal weight
    days_to_goal = (goal_weight - intercept) / slope
    goal_date = data.index.min() + np.timedelta64(int(days_to_goal), 'D')

    # Plot the extrapolated regression line (dashed black)
    future_dates = np.arange(X.max(), days_to_goal+10).reshape(-1, 1)
    y_future = model_weighted.predict(future_dates)
    ax.plot(mdates.num2date(mdates.date2num(data.index.min()) + future_dates.flatten()), y_future, color='black', linestyle='dashed')

    # Plot the vertical line at the goal date
    ax.axvline(goal_date, color='grey', linestyle='dashed', label=f'{lift_type} Goal ({goal_weight} lbs)')

    # Annotate the goal date
    ax.annotate(f'Goal: {goal_weight} lbs\n{str(goal_date)[:10]}',  # Convert goal_date to string and extract date
                xy=(goal_date, goal_weight), xytext=(goal_date, goal_weight *1.05),
                fontsize=10, color=color)

    y_offset = 1.2
    ax.set_ylim(0, goal_weight * y_offset)

    
def plot_body_composition(body_comp):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Update metrics list - now with body fat as percentage
    metrics = [
        ('Weight (lb)', 'Weight'), 
        ('SMM (lb)', 'SMM'), 
        ('PBF / Body Fat (%)', 'Body Fat')
    ]
    axes = [ax] + [ax.twinx() for _ in range(len(metrics)-1)]
    
    # Offset each additional y-axis
    axis_offset = 60  # points
    for i, axis in enumerate(axes[1:], 1):
        axis.spines['right'].set_position(('outward', i * axis_offset))
    
    for (metric, metric_label), axis, color in zip(metrics, axes, ['#1f77b4', '#ff7f0e', '#2ca02c']):
        metric_data = pd.DataFrame({
            'value': pd.to_numeric(body_comp[metric], errors='coerce'),
            'equipment': body_comp['Equipment']
        })
        
        # Drop rows where the metric value is NaN
        metric_data = metric_data.dropna(subset=['value'])
        
        if metric_data.empty:
            continue
            
        # Center the data around the middle of a reasonable range
        mid_point = (metric_data['value'].max() + metric_data['value'].min()) / 2
        range_size = (metric_data['value'].max() - metric_data['value'].min()) * 1.5
        y_min = mid_point - range_size/2
        y_max = mid_point + range_size/2
        
        # Convert dates to numerical values for trend line fitting
        x_num = (metric_data.index - metric_data.index[0]).days
        
        # Plot scattered points with different markers based on equipment
        for eq_type in metric_data['equipment'].unique():
            mask = metric_data['equipment'] == eq_type
            marker = 's' if 'Inbody 270' in str(eq_type) else 'o'
            data_subset = metric_data[mask]
            
            axis.scatter(data_subset.index, data_subset['value'], color=color,
                        marker=marker, edgecolor='black', s=30,
                        label=f'{metric} ({eq_type})')
            
        try:
            # Fit polynomial regression
            poly_model = make_pipeline(
                PolynomialFeatures(degree=3),
                LinearRegression()
            )
            
            # Reshape data for sklearn
            X = np.array(x_num).reshape(-1, 1)
            y = np.array(metric_data['value']).reshape(-1, 1)
            
            # Fit the model
            poly_model.fit(X, y)
            
            # Generate points for smooth curve
            x_smooth = np.linspace(min(x_num), max(x_num), 200).reshape(-1, 1)
            y_smooth = poly_model.predict(x_smooth)
            
            # Convert x_smooth back to dates
            dates_smooth = metric_data.index[0] + pd.to_timedelta(x_smooth.ravel(), unit='D')
            
            # Plot the trend line
            if metric == 'Weight (lb)':
                axis.plot(dates_smooth, y_smooth, '-', color=color, alpha=0.8,
                     label=f'{metric} (trend)')
            else:
                axis.plot(dates_smooth, y_smooth, '--', color=color, alpha=0.8,
                     label=f'{metric} (trend)')
        
        except Exception as e:
            print(f"Warning: Could not fit polynomial curve for {metric}: {str(e)}")
            # Fallback to linear trend
            z = np.polyfit(x_num, metric_data['value'], 1)
            p = np.poly1d(z)
            trend_dates = pd.date_range(metric_data.index[0], metric_data.index[-1], periods=100)
            trend_x_num = (trend_dates - metric_data.index[0]).days
            axis.plot(trend_dates, p(trend_x_num), '--', color=color, alpha=0.8,
                     label=f'{metric} (trend)')
        
        # Set axis limits and label
        axis.set_ylim(y_min, y_max)
        axis.spines['right'].set_color(color)
        axis.tick_params(axis='y', colors=color)
        
        # Calculate and print total change
        total_change = metric_data['value'].iloc[-1] - metric_data['value'].iloc[0]
        print(f'Total change in {metric}: {total_change:.2f}')
        
        # Add change annotation with appropriate unit
        y_pos = metric_data['value'].iloc[-1]
        x_pos = metric_data.index[-1] + pd.Timedelta(days=5)
        if metric == 'PBF / Body Fat (%)':
            change_text = f'{metric_label}:\n$\Delta$ {total_change:+.2f}%'
        else:
            change_text = f'{metric_label}:\n$\Delta$ {total_change:+.2f} lbs'
        axis.text(x_pos, y_pos, change_text,
                 color=color, ha='left', va='center', fontsize=8)
        
        # Add metric label
        axis.set_ylabel(metric, color=color)

    # Add vertical line for "Post meet"
    post_meet_mask = body_comp['Notes'] == 'Post meet'
    if any(post_meet_mask):
        post_meet_date = body_comp.index[post_meet_mask][0]
        ax.axvline(x=post_meet_date, color='black', linestyle='--', alpha=0.8)
        
        # Add "Post meet" label
        ylim = ax.get_ylim()
        label_x = post_meet_date + pd.Timedelta(days=20)  # move 5 days right
        label_y = ylim[1] + (ylim[1] - ylim[0]) * 0.02 - 3
        ax.text(label_x, label_y, 'Post meet', 
                rotation=0, ha='center', va='bottom', color='black')

    # Style the plot
    ax.set_title('Body Composition Metrics', pad=20)
    ax.set_xlabel('Date')
    
    # Format x-axis
    ax.xaxis_date()
    # Label as Month-Year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Create custom legend for equipment types
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                  markeredgecolor='black', markersize=8, label='Inbody 270'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markeredgecolor='black', markersize=8, label='Other Equipment')
    ]
    
    # Add equipment legend to the plot
    ax.legend(handles=legend_elements, loc='upper left', frameon=True)
    
    # Remove metric legends
    for axis in axes[1:]:  # Skip the first axis since it now has our equipment legend
        axis.legend().set_visible(False)

    # Adjust layout to prevent label overlap
    fig.tight_layout()
    plt.show()