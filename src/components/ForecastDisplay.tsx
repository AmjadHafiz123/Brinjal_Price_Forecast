import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"; 
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Coins, Calendar } from "lucide-react";
import { format, addDays } from "date-fns";

interface ForecastData {
  date: string;
  price: number;
  trend: "up" | "down" | "stable";
  confidence: number;
}

interface ForecastDisplayProps {
  vegetable: string;
  mode: "single" | "range";
  singleDate?: Date;
  dateRange?: { from: Date | undefined; to: Date | undefined };
}

const vegetableEmojis: Record<string, string> = {
  brinjal: "🍆",
  cabbage: "🥬",
  beans: "🫘",
};

export function ForecastDisplay({ vegetable, mode, singleDate, dateRange }: ForecastDisplayProps) {
  const [forecasts, setForecasts] = useState<ForecastData[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setForecasts(null)
    if (!vegetable) return;

    let startDate: Date | undefined;
    let endDate: Date | undefined;

    if (mode === "single" && singleDate) {
      startDate = singleDate;
      endDate = singleDate;
    } else if (mode === "range" && dateRange?.from && dateRange?.to) {
      startDate = dateRange.from;
      endDate = dateRange.to;
    }
    if (!startDate || !endDate) return;
    const fetchForecast = async () => {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch("http://127.0.0.1:5000/forecast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            vegetable: vegetable,
            startdate: format(startDate, "yyyy-MM-dd"),
            enddate: format(endDate, "yyyy-MM-dd"),
          }),
        });

        if (!res.ok) throw new Error("Failed to fetch forecast");

        const data: ForecastData[] = await res.json();
        setForecasts(data);
      } catch (err) {
        setError("Failed to load forecast.");
      } finally {
        setLoading(false);
      }
    };

    fetchForecast();
  }, [vegetable, mode, singleDate, dateRange]);

  if (!vegetable) {
    return (
      <Card className="h-96 flex items-center justify-center">
        <CardContent className="text-center">
          <div className="text-6xl mb-4">🥬</div>
          <p className="text-muted-foreground">Select a vegetable to see price forecasts</p>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card className="h-96 flex items-center justify-center">
        <CardContent className="text-center">
          <p>Loading forecast...</p>
        </CardContent>
      </Card>
    );
  }

  if (error || !forecasts) {
    return (
      <Card className="h-96 flex items-center justify-center">
        <CardContent className="text-center">
          <p className="text-red-500">{error || "No data available"}</p>
        </CardContent>
      </Card>
    );
  }

  const avgPrice = forecasts.reduce((sum, f) => sum + f.price, 0) / forecasts.length;
  const avgConfidence = forecasts.reduce((sum, f) => sum + f.confidence, 0) / forecasts.length;

  return (
    <div className="space-y-6 animate-slide-up">
      <Card className="shadow-elegant">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-3">
            <span className="text-3xl">{vegetableEmojis[vegetable] || "🥬"}</span>
            <div>
              <h3 className="text-xl font-semibold capitalize">{vegetable.replace("-", " ")} Forecast</h3>
              <p className="text-sm text-muted-foreground">
                {(mode === "single" && singleDate) && format(singleDate!, "MMMM d, yyyy") }
                {(mode === "range" && dateRange?.from && dateRange?.to) && `${format(dateRange!.from!, "MMM d")} - ${format(dateRange!.to!, "MMM d, yyyy")}`}
              </p>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-gradient-subtle p-4 rounded-lg border">
              <div className="flex items-center gap-2 mb-2">
              <Coins className="h-5 w-5 text-yellow-500" />
                <span className="text-sm font-medium">Average Price</span>
              </div>
              <div className="text-2xl font-bold">{avgPrice.toFixed(2)} Rs</div>
              <p className="text-xs text-muted-foreground">per Kg</p>
            </div>

            <div className="bg-gradient-subtle p-4 rounded-lg border">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Confidence</span>
              </div>
              <div className="text-2xl font-bold">{Math.round(avgConfidence * 100)}%</div>
              <p className="text-xs text-muted-foreground">model accuracy</p>
            </div>

            <div className="bg-gradient-subtle p-4 rounded-lg border">
              <div className="flex items-center gap-2 mb-2">
                <Calendar className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Period</span>
              </div>
              <div className="text-2xl font-bold">{forecasts.length}</div>
              <p className="text-xs text-muted-foreground">{forecasts.length === 1 ? "day" : "days"}</p>
            </div>
          </div>

          {forecasts.length > 1 ? (
            <div className="space-y-3">
              <h4 className="font-medium mb-3">Daily Breakdown</h4>
              <div className="max-h-60 overflow-y-auto space-y-2">
                {forecasts.map((forecast) => (
                  <div
                    key={forecast.date}
                    className="flex items-center justify-between p-3 bg-card border rounded-lg hover:bg-accent/50 transition-colors"
                  >
                    <div>
                      <div className="font-medium">{format(new Date(forecast.date), "MMM d, yyyy")}</div>
                      <div className="text-sm text-muted-foreground">
                        Confidence: {Math.round(forecast.confidence * 100)}%
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="text-right">
                        <div className="font-bold text-lg">{forecast.price} Rs</div>
                        <div className="text-xs text-muted-foreground">per Kg</div>
                      </div>
                      <Badge
                        variant={
                          forecast.trend === "up"
                            ? "default"
                            : forecast.trend === "down"
                            ? "destructive"
                            : "secondary"
                        }
                        className="flex items-center gap-1"
                      >
                        {forecast.trend === "up" && <TrendingUp className="h-3 w-3" />}
                        {forecast.trend === "down" && <TrendingDown className="h-3 w-3" />}
                        {forecast.trend}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center p-6 bg-gradient-subtle rounded-lg border">
              <div className="text-4xl font-bold text-primary mb-2">{forecasts[0].price} Rs</div>
              <div className="text-muted-foreground mb-4">
                per Kg on {format(new Date(forecasts[0].date), "MMMM d, yyyy")}
              </div>
              <Badge
                variant={
                  forecasts[0].trend === "up"
                    ? "default"
                    : forecasts[0].trend === "down"
                    ? "destructive"
                    : "secondary"
                }
                className="flex items-center gap-1 w-fit mx-auto"
              >
                {forecasts[0].trend === "up" && <TrendingUp className="h-3 w-3" />}
                {forecasts[0].trend === "down" && <TrendingDown className="h-3 w-3" />}
                {forecasts[0].trend} trend
              </Badge>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
