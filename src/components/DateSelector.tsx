import * as React from "react";
import { CalendarIcon } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface DateSelectorProps {
  singleDate: Date | undefined;
  dateRange: { from: Date | undefined; to: Date | undefined };
  onSingleDateChange: (date: Date | undefined) => void;
  onDateRangeChange: (range: { from: Date | undefined; to: Date | undefined }) => void;
  mode: "single" | "range";
  onModeChange: (mode: "single" | "range") => void;
}

export function DateSelector({
  singleDate,
  dateRange,
  onSingleDateChange,
  onDateRangeChange,
  mode,
  onModeChange,
}: DateSelectorProps) {
  return (
    <div className="space-y-4">
      <Label className="text-sm font-medium">Forecast Period</Label>
      
      <Tabs value={mode} onValueChange={(value) => onModeChange(value as "single" | "range")}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="single">Single Date</TabsTrigger>
          <TabsTrigger value="range">Date Range</TabsTrigger>
        </TabsList>
        
        <TabsContent value="single" className="mt-4">
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-full h-12 justify-start text-left font-normal",
                  !singleDate && "text-muted-foreground"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {singleDate ? format(singleDate, "PPP") : <span>Pick a date</span>}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={singleDate}
                onSelect={onSingleDateChange}
                initialFocus
                className="p-3 pointer-events-auto"
                disabled={(date) => date < new Date()}
              />
            </PopoverContent>
          </Popover>
        </TabsContent>
        
        <TabsContent value="range" className="mt-4">
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-full h-12 justify-start text-left font-normal",
                  !dateRange.from && "text-muted-foreground"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {dateRange.from ? (
                  dateRange.to ? (
                    <>
                      {format(dateRange.from, "LLL dd, y")} - {format(dateRange.to, "LLL dd, y")}
                    </>
                  ) : (
                    format(dateRange.from, "LLL dd, y")
                  )
                ) : (
                  <span>Pick a date range</span>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="range"
                selected={dateRange.from ? { from: dateRange.from, to: dateRange.to } : undefined}
                onSelect={(range) => {
                  if (range) {
                    onDateRangeChange({
                      from: range.from,
                      to: range.to
                    });
                  } else {
                    onDateRangeChange({ from: undefined, to: undefined });
                  }
                }}
                initialFocus
                className="p-3 pointer-events-auto"
                disabled={(date) => date < new Date()}
                numberOfMonths={2}
              />
            </PopoverContent>
          </Popover>
        </TabsContent>
      </Tabs>
    </div>
  );
}