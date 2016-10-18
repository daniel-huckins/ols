package ols

import (
	"github.com/gonum/matrix/mat64"
)

// Model handles the regression
type Model struct {
	x *mat64.Matrix
	y *mat64.Matrix // one dimentional
}

// NewModel creates a new OLS model
func NewModel(x *mat64.Matrix, y *mat64.Matrix) *Model {
	return &Model{
		x: x, y: y,
	}
}
