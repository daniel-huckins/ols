package ols

import (
	"github.com/gonum/matrix/mat64"
)

// Model handles the regression
type Model struct {
	x *mat64.Dense
	y *mat64.Vector // one dimentional
}
