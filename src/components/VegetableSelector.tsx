import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";

const vegetables = [
 { id: "brinjal", name: "Brinjal", emoji: "🍆" } ,
 { id: "cabbage", name: "Cabbage", emoji: "🥬" },
 { id: "beans", name: "Beans", emoji: "🫘" }
];

interface VegetableSelectorProps {
  value: string;
  onValueChange: (value: string) => void;
}

export function VegetableSelector({ value, onValueChange }: VegetableSelectorProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="vegetable-select" className="text-sm font-medium">
        Select Vegetable
      </Label>
      <Select value={value} onValueChange={onValueChange}>
        <SelectTrigger className="w-full h-12 border-border focus:ring-primary">
          <SelectValue placeholder="Choose a vegetable to forecast" />
        </SelectTrigger>
        <SelectContent className="bg-popover border-border">
          {vegetables.map((vegetable) => (
            <SelectItem 
              key={vegetable.id} 
              value={vegetable.id}
              className="flex items-center gap-2 cursor-pointer hover:bg-accent"
            >
              <span className="text-lg">{vegetable.emoji}</span>
              <span>{vegetable.name}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}