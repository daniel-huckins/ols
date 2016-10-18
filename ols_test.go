package ols

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

// data from https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
func TestOLS(t *testing.T) {
	x := mat64.NewVector(15, []float64{
		1.47, 1.50, 1.52, 1.55, 1.57,
		1.60, 1.63, 1.65, 1.68, 1.70,
		1.73, 1.75, 1.78, 1.80, 1.83,
	})
	y := mat64.NewVector(15, []float64{
		52.21, 53.12, 54.48, 55.84, 57.20,
		58.57, 59.93, 61.29, 63.11, 64.47,
		66.28, 68.10, 69.92, 72.19, 74.46,
	})
	model := NewModel(x, y)
	res, err := model.Train()
	if err != nil {
		t.Fatalf("error in training: %s\n", err.Error())
	}
	t.Logf("%+v\n", res)
}
