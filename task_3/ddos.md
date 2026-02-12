In this task, I performed traffic analysis on a real web server event log in order to detect possible Distributed Denial of Service (DDoS) attacks. A DDoS attack attempts to overwhelm a server by sending an extremely large number of requests in a short period of time, which can disrupt normal service availability. Instead of using simple threshold-based detection, I applied regression analysis to model normal traffic behavior and identify statistically significant anomalies.

The provided log file was uploaded to GitHub in the same folder as the source code:

GitHub link to log file:

task_3/n_gvazava25_24957_server.log 

The objective of this work was to process the log data, construct a regression model representing expected traffic trends, and detect abnormal traffic spikes that indicate DDoS activity. The entire process is fully reproducible using the provided source code and described methodology.



Methodology

First, I parsed the timestamp from each log entry. The timestamps had the format:

[2024-03-22 18:00:09+04:00]

I extracted and converted them using:

TS_REGEX = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\]")

dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S%z")


Then I aggregated requests per minute:

df["bucket"] = df["timestamp"].dt.floor("1min")
traffic = df.groupby("bucket").size().reset_index(name="request_count")


To model normal traffic behavior, I applied Linear Regression:

model = LinearRegression()
model.fit(X, y)
traffic["predicted"] = model.predict(X)
traffic["residual"] = traffic["request_count"] - traffic["predicted"]


Residuals were standardized using Z-score:

traffic["residual_z"] = scaler.fit_transform(traffic[["residual"]])
traffic["is_ddos"] = traffic["residual_z"] > 3.0


Any value above 3 standard deviations was marked as anomalous.

3. Regression Analysis Results

The regression model estimated the normal traffic trend over time. When actual traffic significantly exceeded predicted values, the residuals became large and positive.

The following DDoS intervals were detected:

• 2024-03-22 18:12:00 → 18:13:00 (+04:00)
• 2024-03-22 18:14:00 → 18:16:00 (+04:00)

The second interval reached a peak of 14,811 requests per minute with a Z-score of 4.20, indicating a strong anomaly.

<img width="622" height="466" alt="image" src="https://github.com/user-attachments/assets/952e4820-5b41-4675-87c9-1313c02a7f22" />




Conclusion

This work demonstrates how regression analysis can be effectively applied to cybersecurity log data for anomaly detection. By modeling normal web traffic using Linear Regression and analyzing standardized residuals, I was able to statistically identify abnormal traffic spikes that correspond to DDoS activity.

Two attack intervals were detected, with the strongest attack occurring between 18:14 and 18:16, reaching more than 14,800 requests per minute and exceeding 4 standard deviations above the expected traffic level. Such deviation is highly unlikely under normal conditions, confirming the presence of malicious activity.

The methodology is transparent and reproducible: extract timestamps, aggregate traffic, build a regression model, compute residuals, and detect anomalies using Z-score thresholding. This approach provides a mathematically grounded and reliable way to detect DDoS attacks from raw server logs, and it can be extended to real-time monitoring systems in practical cybersecurity environments.





