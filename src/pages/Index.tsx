import { useState } from "react";
import { VegetableSelector } from "@/components/VegetableSelector";
import { DateSelector } from "@/components/DateSelector";
import { ForecastDisplay } from "@/components/ForecastDisplay";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Leaf, TrendingUp } from "lucide-react";

const Index = () => {
  const [selectedVegetable, setSelectedVegetable] = useState("");
  const [singleDate, setSingleDate] = useState<Date | undefined>();
  const [dateRange, setDateRange] = useState<{ from: Date | undefined; to: Date | undefined }>({
    from: undefined,
    to: undefined,
  });
  const [dateMode, setDateMode] = useState<"single" | "range">("single");

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8 animate-fade-in">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="bg-gradient-primary p-3 rounded-full shadow-glow">
              <Leaf className="h-8 w-8 text-primary-foreground" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              VegiForecast
            </h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Predict vegetable prices with our advanced ML models. Select a vegetable and forecast period to get accurate price predictions.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Control Panel */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="shadow-elegant animate-fade-in">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  Forecast Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <VegetableSelector 
                  value={selectedVegetable} 
                  onValueChange={setSelectedVegetable} 
                />
                <DateSelector
                  singleDate={singleDate}
                  dateRange={dateRange}
                  onSingleDateChange={setSingleDate}
                  onDateRangeChange={setDateRange}
                  mode={dateMode}
                  onModeChange={setDateMode}
                />
              </CardContent>
            </Card>

            {/* ML Model Info */}
            <Card className="shadow-elegant animate-fade-in">
              <CardHeader>
                <CardTitle className="text-sm">About Our Models</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground space-y-2">
                <p>Our ML models use:</p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>Historical price data</li>
                  <li>Seasonal patterns</li>
                  <li>Market demand trends</li>
                </ul>
                <p className="pt-2 text-xs">
                  *Forecasts shown are mock predictions for demonstration purposes.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Forecast Display */}
          <div className="lg:col-span-2">
            <ForecastDisplay
              vegetable={selectedVegetable}
              mode={dateMode}
              singleDate={singleDate}
              dateRange={dateRange}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
